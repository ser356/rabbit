#if canImport(CoreML)
import CoreML
import Foundation
import XCTest
@testable import RabbitSenseKit

/// Swift + Core ML frente a PyTorch en ventanas reales (Fixtures/models, generado en el Mac con
/// `python -m rabbitsense.swiftgolden models <vídeos>`). Se salta si faltan esos datos o los modelos convertidos
/// (RabbitSense_repo/weights/coreml, o la carpeta de la variable RABBITSENSE_WEIGHTS).
final class ModelParityTests: XCTestCase {
    static let here = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
    static let fixtures = here.appendingPathComponent("Fixtures/models")
    static let weights = ProcessInfo.processInfo.environment["RABBITSENSE_WEIGHTS"].map { URL(fileURLWithPath: $0) }
        ?? here.appendingPathComponent("../../../../weights/coreml").standardizedFileURL
    static let units: MLComputeUnits = ProcessInfo.processInfo.environment["RABBITSENSE_COMPUTE"] == "ne"
        ? .cpuAndNeuralEngine : .all

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

    func golden() throws -> Golden {
        let url = Self.fixtures.appendingPathComponent("models.json")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: url.path), "Sin Fixtures/models: ejecuta swiftgolden models")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.weights.appendingPathComponent("modelos.json").path),
                          "Sin modelos convertidos en \(Self.weights.path): ejecuta rabbitsense.coreml convert")
        return try JSONDecoder().decode(Golden.self, from: Data(contentsOf: url))
    }

    func frame(_ window: Golden.Window, _ f: Int) throws -> RGBImage {
        try loadImage(Self.fixtures.appendingPathComponent(window.name).appendingPathComponent(String(format: "%06d.png", f)))
    }

    func testDetectorMatchesPyTorch() throws {
        let data = try golden()
        let detector = try OwlDetector(url: Self.weights.appendingPathComponent("owlv2.mlpackage"), computeUnits: Self.units)
        var same = 0, total = 0
        var seconds: [Double] = []
        for window in data.windows {
            for (f, expected) in zip(window.probe, window.found) {
                let image = try frame(window, f)
                let started = Date()
                let found = try detector(image)
                seconds.append(Date().timeIntervalSince(started))
                total += 1
                if found.isEmpty == expected.isEmpty && (found.isEmpty || iou(found[0], expected[0]) >= 0.9) { same += 1 }
            }
        }
        print("OWLv2 Swift: \(same)/\(total) iguales; \(seconds.sorted()[seconds.count / 2]) s por fotograma")
        XCTAssertGreaterThanOrEqual(Double(same) / Double(total), 0.95)
    }

    func testEncoderAndActMatchPyTorch() throws {
        let data = try golden()
        let encoder = try VJEPAEncoder(url: Self.weights.appendingPathComponent("vjepa2.mlpackage"),
                                       meta: Self.weights.appendingPathComponent("modelos.json"), computeUnits: Self.units)
        let classifier = try ActClassifier.bundled()
        var cosines: [Double] = [], sameAct = 0, windows = 0
        for window in data.windows {
            guard let square = window.square, let expected = window.embedding, let act = window.act else { continue }
            let box = SquareBox(left: square[0], top: square[1], side: square[2])
            let clip = try window.picks.map { squareCrop(try frame(window, $0), box: box, size: encoder.size) }
            let vector = try encoder(clip).map(Double.init)
            let dot = zip(vector, expected).map(*).reduce(0, +)
            let norms = (vector.map { $0 * $0 }.reduce(0, +) * expected.map { $0 * $0 }.reduce(0, +)).squareRoot()
            cosines.append(dot / norms)
            windows += 1
            if classifier.predict(vector.map(Float.init)).act == act { sameAct += 1 }
        }
        try XCTSkipIf(windows == 0, "Ninguna ventana con conejo en los datos de referencia")
        print("V-JEPA 2 Swift: coseno mínimo \(cosines.min()!); mismo acto \(sameAct)/\(windows)")
        XCTAssertGreaterThanOrEqual(cosines.sorted()[cosines.count / 2], 0.99)
        XCTAssertGreaterThanOrEqual(Double(sameAct) / Double(windows), 0.9)
    }

    func testSAM2MatchesPyTorch() throws {
        let data = try golden()
        let folder = Self.weights.appendingPathComponent("sam2")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: folder.path), "Sin SAM2 de Core ML: queryseg download-coreml")
        let sam = try SAM2Segmenter(folder: folder, computeUnits: Self.units)
        var ious: [Double] = []
        for window in data.windows {
            for (f, found) in zip(window.probe, window.found) where window.masks.contains(f) {
                let image = try frame(window, f)
                let reference = try loadImage(Self.fixtures.appendingPathComponent(window.name)
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
        try XCTSkipIf(ious.isEmpty, "Sin máscaras de referencia")
        print("SAM2 Swift: IoU mediana \(ious.sorted()[ious.count / 2]) en \(ious.count) fotogramas")
        XCTAssertGreaterThanOrEqual(ious.sorted()[ious.count / 2], 0.8)
    }
}
#endif
