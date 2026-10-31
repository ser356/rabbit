#if canImport(CoreML)
import CoreML
import CoreGraphics
import CoreVideo
import Foundation
import ImageIO

/// Carga un modelo Core ML (.mlpackage o .mlmodelc), compilándolo si hace falta.
public func loadModel(_ url: URL, computeUnits: MLComputeUnits = .all) throws -> MLModel {
    let compiled = url.pathExtension == "mlmodelc" ? url : try MLModel.compileModel(at: url)
    let configuration = MLModelConfiguration()
    configuration.computeUnits = computeUnits
    return try MLModel(contentsOf: compiled, configuration: configuration)
}

/// Valores de un MLMultiArray como Double, en orden lineal (sea Float16, Float32 o Double).
func values(_ array: MLMultiArray) -> [Double] {
    let n = array.count
    switch array.dataType {
    case .float32:
        let p = array.dataPointer.bindMemory(to: Float.self, capacity: n)
        return (0..<n).map { Double(p[$0]) }
    case .double:
        let p = array.dataPointer.bindMemory(to: Double.self, capacity: n)
        return (0..<n).map { p[$0] }
    default:
        return (0..<n).map { array[$0].doubleValue }
    }
}

/// MLMultiArray Float32 con la forma dada, rellenado con `data`.
func multiArray(_ data: [Float], shape: [Int]) throws -> MLMultiArray {
    let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
    let p = array.dataPointer.bindMemory(to: Float.self, capacity: data.count)
    data.withUnsafeBufferPointer { p.update(from: $0.baseAddress!, count: data.count) }
    return array
}

/// Detector OWLv2 convertido (`owlv2.mlpackage`: entrada «pixels», salidas «logits» y «boxes»).
public final class OwlDetector {
    let model: MLModel
    public let size: Int

    public init(url: URL, size: Int = 960, computeUnits: MLComputeUnits = .all) throws {
        model = try loadModel(url, computeUnits: computeUnits)
        self.size = size
    }

    /// Detecciones en una imagen RGB.
    public func callAsFunction(_ image: RGBImage) throws -> [Detection] {
        let pixels = try multiArray(owlPixels(image, size: size), shape: [1, 3, size, size])
        let output = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["pixels": pixels]))
        guard let logits = output.featureValue(for: "logits")?.multiArrayValue,
              let boxes = output.featureValue(for: "boxes")?.multiArrayValue else {
            throw RabbitSenseError.model("OWLv2 no devolvió logits y boxes")
        }
        let scores = values(logits).map { 1 / (1 + exp(-$0)) }
        return detections(scores: scores, boxes: values(boxes), width: image.width, height: image.height)
    }
}

/// V-JEPA 2 convertido (`vjepa2.mlpackage`: entrada «video» (1, 16, 3, 256, 256), salida «embedding»).
public final class VJEPAEncoder {
    let model: MLModel
    public let size: Int
    public let frames: Int
    let mean: [Float]
    let std: [Float]

    /// - Parameter meta: `modelos.json` junto a los modelos (tamaño, fotogramas, media y desviación).
    public init(url: URL, meta: URL, computeUnits: MLComputeUnits = .all) throws {
        model = try loadModel(url, computeUnits: computeUnits)
        struct Meta: Decodable {
            struct V: Decodable { let size: Int; let frames: Int; let mean: [Float]; let std: [Float] }
            let vjepa: V
        }
        let m = try JSONDecoder().decode(Meta.self, from: Data(contentsOf: meta)).vjepa
        size = m.size
        frames = m.frames
        mean = m.mean
        std = m.std
    }

    /// Vector de la ventana a partir de `frames` recortes RGB de `size` × `size`.
    public func callAsFunction(_ clip: [RGBImage]) throws -> [Float] {
        precondition(clip.count == frames && clip.allSatisfy { $0.width == size && $0.height == size })
        var data = [Float](repeating: 0, count: frames * 3 * size * size)
        for (t, image) in clip.enumerated() {
            for c in 0..<3 {
                let base = (t * 3 + c) * size * size
                for i in 0..<(size * size) {
                    data[base + i] = (Float(image.pixels[i * 3 + c]) / 255 - mean[c]) / std[c]
                }
            }
        }
        let video = try multiArray(data, shape: [1, frames, 3, size, size])
        let output = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["video": video]))
        guard let embedding = output.featureValue(for: "embedding")?.multiArrayValue else {
            throw RabbitSenseError.model("V-JEPA 2 no devolvió embedding")
        }
        return values(embedding).map(Float.init)
    }
}

/// SAM2 de imagen de Core ML publicado por Apple (`apple/coreml-sam2.1-small`): codificador de imagen, de
/// indicaciones y decodificador. Los nombres de entradas y salidas se leen de los propios modelos; la caja se da como
/// dos esquinas con las etiquetas 2 y 3, en coordenadas de la imagen cuadrada de entrada.
public final class SAM2Segmenter {
    let encoder: MLModel
    let prompt: MLModel
    let decoder: MLModel
    let size: Int

    /// - Parameter folder: carpeta con los tres .mlpackage (ImageEncoder, PromptEncoder, MaskDecoder).
    public init(folder: URL, computeUnits: MLComputeUnits = .all) throws {
        func find(_ kind: String) throws -> URL {
            let items = FileManager.default.enumerator(at: folder, includingPropertiesForKeys: nil)?
                .compactMap { $0 as? URL } ?? []
            guard let url = items.filter({ $0.lastPathComponent.contains(kind) && $0.pathExtension == "mlpackage" })
                .sorted(by: { $0.path < $1.path }).first else {
                throw RabbitSenseError.model("Falta el modelo \(kind) de SAM2 en \(folder.path)")
            }
            return url
        }
        encoder = try loadModel(find("ImageEncoder"), computeUnits: computeUnits)
        prompt = try loadModel(find("PromptEncoder"), computeUnits: computeUnits)
        decoder = try loadModel(find("MaskDecoder"), computeUnits: computeUnits)
        let input = encoder.modelDescription.inputDescriptionsByName.values.first!
        size = input.imageConstraint?.pixelsWide ?? 1024
    }

    /// Máscara binaria (0 o 255) del tamaño de la imagen, para la caja (x0, y0, x1, y1).
    public func callAsFunction(_ image: RGBImage, box: [Double]) throws -> [UInt8] {
        let square = resizeArea(image, width: size, height: size)
        let input = encoder.modelDescription.inputDescriptionsByName.values.first!
        let value: MLFeatureValue
        if input.type == .image {
            value = MLFeatureValue(pixelBuffer: try pixelBuffer(square))
        } else {
            value = MLFeatureValue(multiArray: try multiArray(square.pixels.map { Float($0) }, shape: [1, size, size, 3]))
        }
        let encoded = try encoder.prediction(from: MLDictionaryFeatureProvider(dictionary: [input.name: value]))
        let sx = Double(size) / Double(image.width), sy = Double(size) / Double(image.height)
        let names = Array(prompt.modelDescription.inputDescriptionsByName.keys)
        guard let pointsName = names.first(where: { $0.lowercased().contains("point") || $0.lowercased().contains("coord") }),
              let labelsName = names.first(where: { $0 != pointsName }) else {
            throw RabbitSenseError.model("No se reconocen las entradas del codificador de indicaciones: \(names)")
        }
        let points = try multiArray([Float(box[0] * sx), Float(box[1] * sy), Float(box[2] * sx), Float(box[3] * sy)],
                                    shape: [1, 2, 2])
        let labels = try multiArray([2, 3], shape: [1, 2])
        let prompts = try prompt.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            pointsName: points, labelsName: labels]))
        var available: [String: MLFeatureValue] = [:]
        for provider in [encoded, prompts] {
            for name in provider.featureNames { available[name] = provider.featureValue(for: name) }
        }
        let needed = Array(decoder.modelDescription.inputDescriptionsByName.keys)
        let missing = needed.filter { available[$0] == nil }
        guard missing.isEmpty else {
            throw RabbitSenseError.model("El decodificador de SAM2 espera \(missing); hay \(available.keys.sorted())")
        }
        let output = try decoder.prediction(from: MLDictionaryFeatureProvider(
            dictionary: Dictionary(uniqueKeysWithValues: needed.map { ($0, available[$0]!) })))
        let arrays = output.featureNames.compactMap { output.featureValue(for: $0)?.multiArrayValue }
        guard let masks = arrays.first(where: { $0.shape.count == 4 }) else {
            throw RabbitSenseError.model("SAM2 no devolvió máscaras")
        }
        let scores = arrays.first(where: { $0.shape.count == 2 }).map(values) ?? [0]
        let best = scores.indices.max { scores[$0] < scores[$1] } ?? 0
        let k = masks.shape[1].intValue, h = masks.shape[2].intValue, w = masks.shape[3].intValue
        let all = values(masks)
        let low = Array(all[(min(best, k - 1) * h * w)..<((min(best, k - 1) + 1) * h * w)])
        var out = [UInt8](repeating: 0, count: image.width * image.height)
        let xs = bilinearIndices(source: w, target: image.width), ys = bilinearIndices(source: h, target: image.height)
        for dy in 0..<image.height {
            let (y0, y1, ly) = ys[dy]
            for dx in 0..<image.width {
                let (x0, x1, lx) = xs[dx]
                let top = Double(1 - lx) * low[y0 * w + x0] + Double(lx) * low[y0 * w + x1]
                let bottom = Double(1 - lx) * low[y1 * w + x0] + Double(lx) * low[y1 * w + x1]
                if Double(1 - ly) * top + Double(ly) * bottom > 0 { out[dy * image.width + dx] = 255 }
            }
        }
        return out
    }

    func pixelBuffer(_ image: RGBImage) throws -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(nil, image.width, image.height, kCVPixelFormatType_32BGRA, nil, &buffer)
        guard status == kCVReturnSuccess, let buffer else { throw RabbitSenseError.model("No se pudo crear la imagen") }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        let row = CVPixelBufferGetBytesPerRow(buffer)
        let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
        for y in 0..<image.height {
            for x in 0..<image.width {
                let p = base + y * row + x * 4, i = (y * image.width + x) * 3
                p[0] = image.pixels[i + 2]
                p[1] = image.pixels[i + 1]
                p[2] = image.pixels[i]
                p[3] = 255
            }
        }
        return buffer
    }
}

/// Lee una imagen (PNG, JPEG…) como RGBImage en RGB.
public func loadImage(_ url: URL) throws -> RGBImage {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        throw RabbitSenseError.missingResource(url.path)
    }
    return try rgbImage(image)
}

/// CGImage → RGBImage en RGB, sin reescalar.
public func rgbImage(_ image: CGImage) throws -> RGBImage {
    let width = image.width, height = image.height
    var rgba = [UInt8](repeating: 0, count: width * height * 4)
    let drawn = rgba.withUnsafeMutableBytes { buffer -> Bool in
        guard let context = CGContext(data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8,
                                      bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return false }
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return true
    }
    guard drawn else { throw RabbitSenseError.model("No se pudo leer la imagen") }
    var rgb = [UInt8](repeating: 0, count: width * height * 3)
    for i in 0..<(width * height) {
        rgb[i * 3] = rgba[i * 4]
        rgb[i * 3 + 1] = rgba[i * 4 + 1]
        rgb[i * 3 + 2] = rgba[i * 4 + 2]
    }
    return RGBImage(width: width, height: height, pixels: rgb)
}
#endif
