import Foundation
import XCTest
@testable import RabbitSenseKit

/// Compara las partes sin modelos con los resultados del código Python validado (Fixtures/pure.json, generado con
/// `python -m rabbitsense.swiftgolden pure`).
final class PureTests: XCTestCase {
    static let fixtures = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures")

    lazy var golden: [String: Any] = {
        let data = try! Data(contentsOf: Self.fixtures.appendingPathComponent("pure.json"))
        return try! JSONSerialization.jsonObject(with: data) as! [String: Any]
    }()

    func cases(_ name: String) -> [[String: Any]] { golden[name] as! [[String: Any]] }

    func doubles(_ value: Any?) -> [Double] { (value as! [NSNumber]).map(\.doubleValue) }

    func bytes(_ value: Any?) -> [UInt8] { (value as! [NSNumber]).map(\.uint8Value) }

    func testClassifierMatchesPython() throws {
        let classifier = try ActClassifier.bundled()
        for c in cases("classifier") {
            let p = classifier.probabilities(doubles(c["x"]).map(Float.init))
            for (a, b) in zip(p, doubles(c["p"])) { XCTAssertEqual(a, b, accuracy: 1e-5) }
        }
    }

    func testDetectionsMatchPython() {
        for c in cases("detections") {
            let found = detections(scores: doubles(c["scores"]), boxes: doubles(c["boxes"]),
                                   width: (c["width"] as! NSNumber).intValue, height: (c["height"] as! NSNumber).intValue)
            let expected = c["found"] as! [[String: NSNumber]]
            XCTAssertEqual(found.count, expected.count)
            for (a, b) in zip(found, expected) {
                XCTAssertEqual(a.score, b["score"]!.doubleValue, accuracy: 1e-9)
                XCTAssertEqual(a.x0, b["x0"]!.doubleValue, accuracy: 1e-6)
                XCTAssertEqual(a.y0, b["y0"]!.doubleValue, accuracy: 1e-6)
                XCTAssertEqual(a.x1, b["x1"]!.doubleValue, accuracy: 1e-6)
                XCTAssertEqual(a.y1, b["y1"]!.doubleValue, accuracy: 1e-6)
            }
        }
    }

    func testWindowSummaryMatchesPython() {
        for c in cases("windows") {
            let perFrame = (c["found"] as! [[[String: NSNumber]]]).map { frame in
                frame.map { Detection(score: $0["score"]!.doubleValue, x0: $0["x0"]!.doubleValue, y0: $0["y0"]!.doubleValue,
                                      x1: $0["x1"]!.doubleValue, y1: $0["y1"]!.doubleValue) }
            }
            let summary = WindowDetection(perFrame)
            XCTAssertEqual(summary.found, (c["count"] as! NSNumber).intValue)
            XCTAssertEqual(summary.probed, (c["probed"] as! NSNumber).intValue)
            XCTAssertEqual(summary.multiple, (c["multiple"] as! NSNumber).boolValue)
            if let union = c["union"] as? [NSNumber] {
                for (a, b) in zip(summary.union!, union) { XCTAssertEqual(a, b.doubleValue, accuracy: 1e-9) }
            } else {
                XCTAssertNil(summary.union)
            }
        }
    }

    func testSquareAroundMatchesPython() {
        for c in cases("square_around") {
            let box = doubles(c["box"]), square = (c["square"] as! [NSNumber]).map(\.intValue)
            let got = squareAround(x0: box[0], y0: box[1], x1: box[2], y1: box[3],
                                   width: (c["width"] as! NSNumber).intValue, height: (c["height"] as! NSNumber).intValue)
            XCTAssertEqual(got, SquareBox(left: square[0], top: square[1], side: square[2]))
        }
    }

    func testProbeIndicesMatchPython() {
        for c in cases("probe") {
            XCTAssertEqual(probeIndices(count: (c["count"] as! NSNumber).intValue, n: (c["n"] as! NSNumber).intValue),
                           (c["indices"] as! [NSNumber]).map(\.intValue))
        }
    }

    func testSquareCropMatchesOpenCV() {
        for c in cases("crops") {
            let image = RGBImage(width: (c["width"] as! NSNumber).intValue, height: (c["height"] as! NSNumber).intValue,
                                 pixels: bytes(c["pixels"]))
            let box = (c["box"] as? [NSNumber]).map { SquareBox(left: $0[0].intValue, top: $0[1].intValue, side: $0[2].intValue) }
            let got = squareCrop(image, box: box, size: (c["size"] as! NSNumber).intValue)
            let expected = bytes(c["out"])
            XCTAssertEqual(got.pixels.count, expected.count)
            let worst = zip(got.pixels, expected).map { abs(Int($0) - Int($1)) }.max() ?? 0
            XCTAssertLessThanOrEqual(worst, 1, "OpenCV redondea en punto fijo al ampliar: ±1 como mucho")
        }
    }

    func testOwlPreprocessingMatchesTransformers() {
        for c in cases("owl") {
            let image = RGBImage(width: (c["width"] as! NSNumber).intValue, height: (c["height"] as! NSNumber).intValue,
                                 pixels: bytes(c["rgb"]))
            let got = owlPixels(image, size: (c["size"] as! NSNumber).intValue)
            let expected = doubles(c["pixels"])
            XCTAssertEqual(got.count, expected.count)
            let worst = zip(got, expected).map { abs(Double($0) - $1) }.max() ?? 0
            XCTAssertLessThan(worst, 1e-4)
        }
    }

    func testFixedCameraDisplacementMatchesPython() {
        for c in cases("fixedcam") {
            let times = doubles(c["times"]), ys = doubles(c["ys"]), areas = doubles(c["areas"])
            let xs = (c["xs"] as! [Any]).map { ($0 as? NSNumber)?.doubleValue }
            let samples = times.indices.map { i in
                MaskSample(time: times[i], x: xs[i], y: xs[i] == nil ? nil : ys[i], area: areas[i])
            }
            let got = FixedCamera.netDisplacement(samples, start: (c["start"] as! NSNumber).doubleValue,
                                                  end: (c["end"] as! NSNumber).doubleValue)
            if let expected = c["displacement"] as? NSNumber {
                XCTAssertEqual(got!, expected.doubleValue, accuracy: 1e-9)
            } else {
                XCTAssertNil(got)
            }
        }
    }

    func testWindowAnalyzerFlow() throws {
        let classifier = try ActClassifier.bundled()
        var frames = [RGBImage](repeating: RGBImage(width: 64, height: 36), count: 30)
        let empty = WindowAnalyzer(detect: { _ in [] }, embed: { _ in [] }, classifier: classifier)
        XCTAssertEqual(try empty.analyze(frames: frames, start: 0, end: 2).act, "not_visible")
        frames[0].pixels[0] = 255
        var clips: [[RGBImage]] = []
        let analyzer = WindowAnalyzer(
            detect: { _ in [Detection(score: 0.9, x0: 10, y0: 5, x1: 30, y1: 20)] },
            embed: { clip in clips.append(clip); return [Float](repeating: 0, count: classifier.dimension) },
            classifier: classifier)
        let result = try analyzer.analyze(frames: frames, start: 2, end: 4)
        XCTAssertEqual(result.found, 4)
        XCTAssertEqual(result.box, squareAround(x0: 10, y0: 5, x1: 30, y1: 20, width: 64, height: 36))
        XCTAssertEqual(clips.first?.count, 16)
        XCTAssertEqual(clips.first?.first?.width, 256)
        XCTAssertTrue(classifier.classes.contains(result.act))
        XCTAssertEqual(result.probabilities.values.reduce(0, +), 1, accuracy: 1e-9)
    }
}
