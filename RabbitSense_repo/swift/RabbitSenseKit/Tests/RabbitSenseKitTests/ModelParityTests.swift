#if canImport(CoreML)
import CoreML
import Foundation
import Testing
@testable import RabbitSenseKit

/// Datos y modelos para comparar Swift + Core ML con PyTorch en ventanas reales.
/// - Fixtures/models: `python -m rabbitsense.swiftgolden models <vídeos>` (en el Mac).
/// - Modelos convertidos: RabbitSense_repo/weights/coreml, o la carpeta de la variable RABBITSENSE_WEIGHTS.
enum Parity {
    static var models: URL { fixtures.appendingPathComponent("models") }
    static var weights: URL {
        ProcessInfo.processInfo.environment["RABBITSENSE_WEIGHTS"].map { URL(fileURLWithPath: $0) }
            ?? fixtures.appendingPathComponent("../../../../../weights/coreml").standardizedFileURL
    }
    static var units: MLComputeUnits {
        ProcessInfo.processInfo.environment["RABBITSENSE_COMPUTE"] == "ne" ? .cpuAndNeuralEngine : .all
    }
    static var available: Bool {
        FileManager.default.fileExists(atPath: models.appendingPathComponent("models.json").path) &&
            FileManager.default.fileExists(atPath: weights.appendingPathComponent("modelos.json").path)
    }
    static var samAvailable: Bool {
        available && FileManager.default.fileExists(atPath: weights.appendingPathComponent("sam2").path)
    }

    struct Golden: Decodable {
        struct Window: Decodable {
            let name: String
            let probe: [Int]
            let picks: [Int]
            let found: [[Detection]]
            let square: [Int]?
            let embedding: [Double]?
            let act: String?
            let masks: [Int]
        }
        let windows: [Window]
    }

    static func golden() throws -> Golden {
        try JSONDecoder().decode(Golden.self, from: Data(contentsOf: models.appendingPathComponent("models.json")))
    }

    static func frame(_ window: Golden.Window, _ f: Int) throws -> RGBImage {
        try loadImage(models.appendingPathComponent(window.name).appendingPathComponent(String(format: "%06d.png", f)))
    }
}

@Suite(.enabled(if: Parity.available, "Faltan Fixtures/models o los modelos convertidos (ver README)"))
struct ModelParityTests {
    @Test func detectorMatchesPyTorch() throws {
        let data = try Parity.golden()
        let detector = try OwlDetector(url: Parity.weights.appendingPathComponent("owlv2.mlpackage"), computeUnits: Parity.units)
        var same = 0, total = 0
        var seconds: [Double] = []
        for window in data.windows {
            for (f, expected) in zip(window.probe, window.found) {
                let image = try Parity.frame(window, f)
                let started = Date()
                let found = try detector(image)
                seconds.append(Date().timeIntervalSince(started))
                total += 1
                if found.isEmpty == expected.isEmpty && (found.isEmpty || iou(found[0], expected[0]) >= 0.9) {
                    same += 1
                } else {
                    func describe(_ d: [Detection]) -> String {
                        d.first.map { String(format: "%.3f [%.0f %.0f %.0f %.0f]", $0.score, $0.x0, $0.y0, $0.x1, $0.y1) }
                            ?? "nada"
                    }
                    let overlap = (found.first != nil && expected.first != nil) ? iou(found[0], expected[0]) : 0
                    print("  distinto \(window.name)/\(f): Python \(describe(expected)) | Swift \(describe(found)) | " +
                          String(format: "IoU %.3f", overlap))
                }
            }
        }
        print("OWLv2 Swift: \(same)/\(total) iguales; \(seconds.sorted()[seconds.count / 2]) s por fotograma")
        #expect(Double(same) / Double(total) >= 0.95)
    }

    @Test func encoderAndActMatchPyTorch() throws {
        let data = try Parity.golden()
        let encoder = try VJEPAEncoder(url: Parity.weights.appendingPathComponent("vjepa2.mlpackage"),
                                       meta: Parity.weights.appendingPathComponent("modelos.json"), computeUnits: Parity.units)
        let classifier = try ActClassifier.bundled()
        var cosines: [Double] = [], sameAct = 0, windows = 0
        var seconds: [Double] = []
        for window in data.windows {
            guard let square = window.square, let expected = window.embedding, let act = window.act else { continue }
            let box = SquareBox(left: square[0], top: square[1], side: square[2])
            let clip = try window.picks.map { squareCrop(try Parity.frame(window, $0), box: box, size: encoder.size) }
            let started = Date()
            let vector = try encoder(clip).map(Double.init)
            seconds.append(Date().timeIntervalSince(started))
            let dot = zip(vector, expected).map(*).reduce(0, +)
            let norms = (vector.map { $0 * $0 }.reduce(0, +) * expected.map { $0 * $0 }.reduce(0, +)).squareRoot()
            cosines.append(dot / norms)
            windows += 1
            if classifier.predict(vector.map(Float.init)).act == act { sameAct += 1 }
        }
        try #require(windows > 0, "Ninguna ventana con conejo en los datos de referencia")
        print("V-JEPA 2 Swift: coseno mínimo \(cosines.min()!); mismo acto \(sameAct)/\(windows); " +
              "\(seconds.sorted()[seconds.count / 2]) s por ventana")
        #expect(cosines.sorted()[cosines.count / 2] >= 0.99)
        #expect(Double(sameAct) / Double(windows) >= 0.9)
    }

    @Test(.enabled(if: Parity.samAvailable, "Sin SAM2 de Core ML: queryseg download-coreml"))
    func sam2MatchesPyTorch() throws {
        let data = try Parity.golden()
        let sam = try SAM2Segmenter(folder: Parity.weights.appendingPathComponent("sam2"), computeUnits: Parity.units)
        var ious: [Double] = []
        for window in data.windows {
            for (f, found) in zip(window.probe, window.found) where window.masks.contains(f) {
                let image = try Parity.frame(window, f)
                let reference = try loadImage(Parity.models.appendingPathComponent(window.name)
                    .appendingPathComponent(String(format: "mask_%06d.png", f)))
                let best = found[0]
                let mask = try sam(image, box: [best.x0, best.y0, best.x1, best.y1])
                var both = 0, either = 0
                for i in 0..<(image.width * image.height) {
                    let a = mask[i] > 0, b = reference.pixels[i * 3] > 127
                    if a && b { both += 1 }
                    if a || b { either += 1 }
                }
                ious.append(either > 0 ? Double(both) / Double(either) : 1)
            }
        }
        try #require(!ious.isEmpty, "Sin máscaras de referencia")
        print("SAM2 Swift: IoU mediana \(ious.sorted()[ious.count / 2]) en \(ious.count) fotogramas")
        #expect(ious.sorted()[ious.count / 2] >= 0.8)
    }
}
#endif
