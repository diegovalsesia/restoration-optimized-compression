/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import ImageProcessing
import CompressorNet
import SwiftUI

enum Mode: String, CaseIterable {
  case xnnpack = "XNNPACK"
  //case coreML = "Core ML"
  //case mps = "MPS"
}

class ProcessingController: ObservableObject {
  @AppStorage("mode") var mode: Mode = .xnnpack
  @Published var elapsedTime: TimeInterval = 0.0
  @Published var isRunning = false

  private let queue = DispatchQueue(label: "org.pytorch.executorch.demo", qos: .userInitiated)
  private var compressor: ImageProcessing?
  private var currentMode: Mode = .xnnpack

    func compress(_ image: UIImage, _ depth: UIImage) {
    guard !isRunning else {
      print("Dropping frame")
      return
    }
    isRunning = true

    if currentMode != mode {
      currentMode = mode
        compressor = nil
    }
    queue.async {
      var returnString: String = ""
      var elapsedTime: TimeInterval = -1
      do {
        if self.compressor == nil {
          self.compressor = try self.createCompressor(for: self.currentMode)
        }
        let startTime = CFAbsoluteTimeGetCurrent()
          returnString = try self.compressor?.compress(image: image, depth: depth) ?? ""
        elapsedTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
      } catch {
        print("Error classifying image: \(error)")
      }
      DispatchQueue.main.async {
        self.elapsedTime = elapsedTime
        self.isRunning = false
      }
    }
  }

  private func createCompressor(for mode: Mode) throws -> ImageProcessing? {
    let modelFileName: String
      switch mode {
          //case .coreML:
          //  modelFileName = "mv3_coreml_all"
          //case .mps:
          //  modelFileName = "mv3_mps_float16"
      case .xnnpack:
          //  modelFileName = "mv3_xnnpack_fp32"
          modelFileName = "compressor"
      }
    guard let modelFilePath = Bundle.main.path(forResource: modelFileName, ofType: "pte")
    else { return nil }
    return try CompressorNet(modelFilePath: modelFilePath)
  }
}
