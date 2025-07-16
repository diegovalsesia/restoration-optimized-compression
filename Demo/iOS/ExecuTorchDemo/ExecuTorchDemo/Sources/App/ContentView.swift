import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var processingController = ProcessingController()
    @State private var uiImage_blur: UIImage?
    @State private var uiImage_depth: UIImage?
    @State private var isImporting = false

    var body: some View {
        VStack {
            if let img = uiImage_blur {
                Image(uiImage: img)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .padding()
            } else {
                Text("No image selected")
                    .foregroundColor(.gray)
            }

            ProcessingTimeView(controller: processingController)

            Button("Select Image") {
                isImporting = true
            }
            .padding()
            .fileImporter(isPresented: $isImporting,
                          allowedContentTypes: [.image],
                          onCompletion: handleImport)
        }
        .onAppear {
            loadImageFromResources()
            classifyCurrentImage()
        }
    }

    private func handleImport(result: Result<URL, Error>) {
        switch result {
        case .success(let url):
            loadImage(from: url)
        case .failure(let error):
            print("Failed to import file: \(error.localizedDescription)")
        }
    }

    private func loadImage(from url: URL) {
        do {
            let data = try Data(contentsOf: url)
            if let image = UIImage(data: data) {
                uiImage_blur = image
                classifyCurrentImage()
            } else {
                print("Failed to create image from data")
            }
        } catch {
            print("Error loading image data: \(error.localizedDescription)")
        }
    }

    private func loadImageFromResources() {
        if let imagePath = Bundle.main.path(forResource: "blur", ofType: "png"),
           let image = UIImage(contentsOfFile: imagePath) {
            uiImage_blur = image
        } else {
            print("Image not found in Resources/Examples directory")
        }
        if let depthPath = Bundle.main.path(forResource: "depth", ofType: "png"),
           let depth = UIImage(contentsOfFile: depthPath) {
            uiImage_depth = depth
        } else {
            print("Depth not found in Resources/Examples directory")
        }
    }

    private func classifyCurrentImage() {
        guard let img = uiImage_blur else { return }
        guard let dep = uiImage_depth else { return }
        processingController.compress(img,dep)
    }
}
