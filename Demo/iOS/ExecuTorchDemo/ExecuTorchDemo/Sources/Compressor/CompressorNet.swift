/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import ExecuTorch
import ImageProcessing
import UIKit

import os.log

import RansLib

public enum CompressorNetError: Error {
  case rawData
  case transform

  var localizedDescription: String {
    switch self {
    case .rawData:
      return "Cannot get the pixel data from the image"
    case .transform:
      return "Cannot transform the image"
    }
  }
}


public class CompressorNet: ImageProcessing {
  private static let resizeSize: CGFloat = 256
  private static let cropSize: CGFloat = 256

  private var module: Module
  private var labels: [String] = []
  private var rawDataBuffer_img: [UInt8]
  private var normalizedBuffer_img: [Float]
  private var rawDataBuffer_depth: [UInt8]
  private var normalizedBuffer_depth: [Float]
  private var lambdaBuffer: [Float]
  private var debugBuffer: [Int32]

  public init?(modelFilePath: String) throws {

    module = Module(filePath: modelFilePath)
      
    rawDataBuffer_img = [UInt8](repeating: 0, count: Int(Self.cropSize * Self.cropSize) * 4)
    normalizedBuffer_img = [Float](repeating: 0, count: rawDataBuffer_img.count / 4 * 3)
    rawDataBuffer_depth = [UInt8](repeating: 0, count: Int(Self.cropSize * Self.cropSize) * 4)
    normalizedBuffer_depth = [Float](repeating: 0, count: rawDataBuffer_depth.count / 4 )
      
    lambdaBuffer = [Float](repeating: 0.0001, count: normalizedBuffer_depth.count)
      
    debugBuffer = [Int32](repeating: -4747, count: normalizedBuffer_depth.count)
      
    #if DEBUG
    Log.shared.add(sink: self)
    #endif
  }

  deinit {
    #if DEBUG
    Log.shared.remove(sink: self)
    #endif
  }
    
    public func compress(image: UIImage, depth: UIImage) throws -> String {
        try normalize_img(transformed(image))
        try normalize_depth(transformed(depth))
        
        let input_img = Tensor<Float>(&normalizedBuffer_img, shape: [1, 3, 256, 256])
        let input_depth = Tensor<Float>(&normalizedBuffer_depth, shape: [1, 1, 256, 256])
        let input_lambda = Tensor<Float>(&lambdaBuffer, shape: [1, 1, 256, 256])
        let debug_tensor = Tensor<Int32>(&debugBuffer, shape: [1, 1, 256, 256])
        
        //let outputs: Tensor<Float> = try module.forward(input_img)[0].tensor()!
        //let outputs = try module.forward(input_img)
        let outputs = try module.forward([input_img,input_lambda,input_depth])

        var encoder = RansEncoder()
        
        let z_symbols : Tensor<Int32> = outputs[0].tensor() ?? debug_tensor
        let z_symbols_sc : [Int32] = try z_symbols.scalars()
        let z_symbols_scvec = Int32Vector(z_symbols_sc)
        
        let z_indexes : Tensor<Int32> = outputs[1].tensor() ?? debug_tensor
        let z_indexes_sc : [Int32] = try z_indexes.scalars()
        let z_indexes_scvec = Int32Vector(z_indexes_sc)
        
        let eb_quantized_cdf : Tensor<Int32> = outputs[2].tensor() ?? debug_tensor
        let eb_quantized_cdf_sc : [Int32] = try eb_quantized_cdf.scalars()
        let swiftNested = stride(from: 0, to: eb_quantized_cdf_sc.count, by: 182).map {
            Array(eb_quantized_cdf_sc[$0..<min($0+182, eb_quantized_cdf_sc.count)])
        }
        let innerVectors: [Int32Vector] = swiftNested.map {Int32Vector($0)}
        let eb_quantized_cdf_scvec = Int32Vector2(innerVectors)
        
        let eb_cdf_length : Tensor<Int32> = outputs[3].tensor() ?? debug_tensor
        let eb_cdf_length_sc : [Int32] = try eb_cdf_length.scalars()
        let eb_cdf_length_scvec = Int32Vector(eb_cdf_length_sc)
        
        let eb_offset : Tensor<Int32> = outputs[4].tensor() ?? debug_tensor
        let eb_offset_sc : [Int32] = try eb_offset.scalars()
        let eb_offset_scvec = Int32Vector(eb_offset_sc)
        
        let y_symbols : Tensor<Int32> = outputs[5].tensor() ?? debug_tensor
        let y_symbols_sc : [Int32] = try y_symbols.scalars()
        let y_symbols_scvec = Int32Vector(y_symbols_sc)
        
        let y_indexes : Tensor<Int32> = outputs[6].tensor() ?? debug_tensor
        let y_indexes_sc : [Int32] = try y_indexes.scalars()
        let y_indexes_scvec = Int32Vector(y_indexes_sc)
        
        let means_hat = outputs[7].tensor() ?? input_img
        
        let gc_quantized_cdf : Tensor<Int32> = outputs[8].tensor() ?? debug_tensor
        let gc_quantized_cdf_sc : [Int32] = try gc_quantized_cdf.scalars()
        let gcswiftNested = stride(from: 0, to: gc_quantized_cdf_sc.count, by: 3133).map {
            Array(gc_quantized_cdf_sc[$0..<min($0+3133, gc_quantized_cdf_sc.count)])
        }
        let gcinnerVectors: [Int32Vector] = gcswiftNested.map {Int32Vector($0)}
        let gc_quantized_cdf_scvec = Int32Vector2(gcinnerVectors)
        
        let gc_cdf_length : Tensor<Int32> = outputs[9].tensor() ?? debug_tensor
        let gc_cdf_length_sc : [Int32] = try gc_cdf_length.scalars()
        let gc_cdf_length_scvec = Int32Vector(gc_cdf_length_sc)
        
        let gc_offset : Tensor<Int32> = outputs[10].tensor() ?? debug_tensor
        let gc_offset_sc : [Int32] = try gc_offset.scalars()
        let gc_offset_scvec = Int32Vector(gc_offset_sc)
        
        let z_strings = encoder.encode_with_indexes(z_symbols_scvec, z_indexes_scvec, eb_quantized_cdf_scvec, eb_cdf_length_scvec, eb_offset_scvec)
        let y_strings = encoder.encode_with_indexes(y_symbols_scvec, y_indexes_scvec, gc_quantized_cdf_scvec, gc_cdf_length_scvec, gc_offset_scvec)
        
        var zz : [UInt8] = Array(repeating: 0, count: z_strings.count)
        for i in 0..<z_strings.count {
            zz[i] = UInt8(z_strings[i])
        }
        saveBytesToFile(bytes: zz, fileName: "z_strings.bin")
        
        var yy : [UInt8] = Array(repeating: 0, count: y_strings.count)
        for i in 0..<y_strings.count {
            yy[i] = UInt8(y_strings[i])
        }
        saveBytesToFile(bytes: yy, fileName: "y_strings.bin")
        
        


        return "Finished"
  }


  private func transformed(_ image: UIImage) throws -> UIImage {
    let aspectRatio = image.size.width / image.size.height
    let targetSize =
      aspectRatio > 1
      ? CGSize(width: Self.resizeSize * aspectRatio, height: Self.resizeSize)
      : CGSize(width: Self.resizeSize, height: Self.resizeSize / aspectRatio)
    let cropRect = CGRect(
      x: (targetSize.width - Self.cropSize) / 2,
      y: (targetSize.height - Self.cropSize) / 2,
      width: Self.cropSize,
      height: Self.cropSize)

    UIGraphicsBeginImageContextWithOptions(cropRect.size, false, 1)
    defer { UIGraphicsEndImageContext() }
    image.draw(
      in: CGRect(
        x: -cropRect.origin.x,
        y: -cropRect.origin.y,
        width: targetSize.width,
        height: targetSize.height))
    guard let resizedAndCroppedImage = UIGraphicsGetImageFromCurrentImageContext()
    else {
      throw CompressorNetError.transform
    }
    return resizedAndCroppedImage
  }

  private func normalize_img(_ image: UIImage) throws {
    guard let cgImage = image.cgImage else {
      throw CompressorNetError.rawData
    }
    let context = CGContext(
      data: &rawDataBuffer_img,
      width: cgImage.width,
      height: cgImage.height,
      bitsPerComponent: 8,
      bytesPerRow: cgImage.width * 4,
      space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
    )
    context?.draw(
      cgImage,
      in: CGRect(
        origin: CGPoint.zero,
        size: CGSize(width: cgImage.width, height: cgImage.height)))

    let pixelCount = rawDataBuffer_img.count / 4

    for i in 0..<pixelCount {
      normalizedBuffer_img[i] = Float(rawDataBuffer_img[i * 4 + 0]) / 255
      normalizedBuffer_img[i + pixelCount] = Float(rawDataBuffer_img[i * 4 + 1]) / 255
      normalizedBuffer_img[i + pixelCount * 2] = Float(rawDataBuffer_img[i * 4 + 2]) / 255
    }
  }
    
    private func normalize_depth(_ depth: UIImage) throws {
      guard let cgImage = depth.cgImage else {
        throw CompressorNetError.rawData
      }
        
      let context = CGContext(
        data: &rawDataBuffer_depth,
        width: cgImage.width,
        height: cgImage.height,
        bitsPerComponent: 8,
        bytesPerRow: cgImage.width * 4,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
      )
      context?.draw(
        cgImage,
        in: CGRect(
          origin: CGPoint.zero,
          size: CGSize(width: cgImage.width, height: cgImage.height)))
      let pixelCount = rawDataBuffer_depth.count / 4

      for i in 0..<pixelCount {
        normalizedBuffer_depth[i] = Float(rawDataBuffer_depth[i * 4 + 0]) / 255
      }
    }
    
    
    func saveBytesToFile(bytes: [UInt8], fileName: String) {
        // Convert [UInt8] to Data
        let data = Data(bytes)
        
        // Get the documents directory URL
        if let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            let fileURL = documentsDirectory.appendingPathComponent(fileName)
            
            do {
                // Write data to file
                try data.write(to: fileURL)
                print("File saved successfully at \(fileURL.path)")
            } catch {
                print("Failed to save file: \(error)")
            }
        } else {
            print("Could not find documents directory")
        }
    }
      
    
}

#if DEBUG
extension CompressorNet: LogSink {
  public func log(level: LogLevel, timestamp: TimeInterval, filename: String, line: UInt, message: String) {
    let logMessage = "executorch:\(filename):\(line) \(message)"

    switch level {
    case .debug:
      os_log(.debug, "%{public}@", logMessage)
    case .info:
      os_log(.info, "%{public}@", logMessage)
    case .error:
      os_log(.error, "%{public}@", logMessage)
    case .fatal:
      os_log(.fault, "%{public}@", logMessage)
    default:
      os_log("%{public}@", logMessage)
    }
  }
}
#endif



func getDocumentsDirectory() -> URL {
    let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
    return paths[0]
}


