import Foundation
import Testing
@testable import RabbitSenseKit

/// Carpeta de datos de referencia, junto a este archivo.
let fixtures = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures")

/// Casos de Fixtures/pure.json (generado con `python -m rabbitsense.swiftgolden pure` desde el código validado).
func cases(_ name: String) throws -> [[String: Any]] {
    let data = try Data(contentsOf: fixtures.appendingPathComponent("pure.json"))
    let golden = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
    return try #require(golden[name] as? [[String: Any]])
}

func doubles(_ value: Any?) -> [Double] { (value as! [NSNumber]).map(\.doubleValue) }
func bytes(_ value: Any?) -> [UInt8] { (value as! [NSNumber]).map(\.uint8Value) }
func int(_ value: Any?) -> Int { (value as! NSNumber).intValue }
func number(_ value: Any?) -> Double { (value as! NSNumber).doubleValue }

/// Partes sin modelos frente a los resultados del código Python validado.
struct PureTests {
    @Test func classifierMatchesPython() throws {
        let classifier = try ActClassifier.bundled()
        for c in try cases("classifier") {
            let p = classifier.probabilities(doubles(c["x"]).map(Float.init))
            for (a, b) in zip(p, doubles(c["p"])) { #expect(abs(a - b) < 1e-5) }
        }
    }

    @Test func detectionsMatchPython() throws {
        for c in try cases("detections") {
            let found = detections(scores: doubles(c["scores"]), boxes: doubles(c["boxes"]),
                                   width: int(c["width"]), height: int(c["height"]))
            let expected = c["found"] as! [[String: NSNumber]]
            #expect(found.count == expected.count)
            for (a, b) in zip(found, expected) {
                #expect(abs(a.score - b["score"]!.doubleValue) < 1e-9)
                #expect(abs(a.x0 - b["x0"]!.doubleValue) < 1e-6)
                #expect(abs(a.y0 - b["y0"]!.doubleValue) < 1e-6)
                #expect(abs(a.x1 - b["x1"]!.doubleValue) < 1e-6)
                #expect(abs(a.y1 - b["y1"]!.doubleValue) < 1e-6)
            }
        }
    }

    @Test func windowSummaryMatchesPython() throws {
        for c in try cases("windows") {
            let perFrame = (c["found"] as! [[[String: NSNumber]]]).map { frame in
                frame.map { Detection(score: $0["score"]!.doubleValue, x0: $0["x0"]!.doubleValue, y0: $0["y0"]!.doubleValue,
                                      x1: $0["x1"]!.doubleValue, y1: $0["y1"]!.doubleValue) }
            }
            let summary = WindowDetection(perFrame)
            #expect(summary.found == int(c["count"]))
            #expect(summary.probed == int(c["probed"]))
            #expect(summary.multiple == (c["multiple"] as! NSNumber).boolValue)
            if let union = c["union"] as? [NSNumber] {
                let got = try #require(summary.union)
                for (a, b) in zip(got, union) { #expect(abs(a - b.doubleValue) < 1e-9) }
            } else {
                #expect(summary.union == nil)
            }
        }
    }

    @Test func squareAroundMatchesPython() throws {
        for c in try cases("square_around") {
            let box = doubles(c["box"]), square = (c["square"] as! [NSNumber]).map(\.intValue)
            let got = squareAround(x0: box[0], y0: box[1], x1: box[2], y1: box[3],
                                   width: int(c["width"]), height: int(c["height"]))
            #expect(got == SquareBox(left: square[0], top: square[1], side: square[2]))
        }
    }

    @Test func probeIndicesMatchPython() throws {
        for c in try cases("probe") {
            #expect(probeIndices(count: int(c["count"]), n: int(c["n"])) == (c["indices"] as! [NSNumber]).map(\.intValue))
        }
    }

    @Test func squareCropMatchesOpenCV() throws {
        for c in try cases("crops") {
            let image = RGBImage(width: int(c["width"]), height: int(c["height"]), pixels: bytes(c["pixels"]))
            let box = (c["box"] as? [NSNumber]).map { SquareBox(left: $0[0].intValue, top: $0[1].intValue, side: $0[2].intValue) }
            let got = squareCrop(image, box: box, size: int(c["size"]))
            let expected = bytes(c["out"])
            #expect(got.pixels.count == expected.count)
            let worst = zip(got.pixels, expected).map { abs(Int($0) - Int($1)) }.max() ?? 0
            #expect(worst <= 1, "OpenCV redondea en punto fijo al ampliar: ±1 como mucho")
        }
    }

    @Test func owlPreprocessingMatchesTransformers() throws {
        for c in try cases("owl") {
            let image = RGBImage(width: int(c["width"]), height: int(c["height"]), pixels: bytes(c["rgb"]))
            let got = owlPixels(image, size: int(c["size"]))
            let expected = doubles(c["pixels"])
            #expect(got.count == expected.count)
            let worst = zip(got, expected).map { abs(Double($0) - $1) }.max() ?? 0
            #expect(worst < 1e-4)
        }
    }

    @Test func fixedCameraDisplacementMatchesPython() throws {
        for c in try cases("fixedcam") {
            let times = doubles(c["times"]), ys = doubles(c["ys"]), areas = doubles(c["areas"])
            let xs = (c["xs"] as! [Any]).map { ($0 as? NSNumber)?.doubleValue }
            let samples = times.indices.map { i in
                MaskSample(time: times[i], x: xs[i], y: xs[i] == nil ? nil : ys[i], area: areas[i])
            }
            let got = FixedCamera.netDisplacement(samples, start: number(c["start"]), end: number(c["end"]))
            if let expected = c["displacement"] as? NSNumber {
                let value = try #require(got)
                #expect(abs(value - expected.doubleValue) < 1e-9)
            } else {
                #expect(got == nil)
            }
        }
    }

    @Test func windowAnalyzerFlow() throws {
        let classifier = try ActClassifier.bundled()
        var frames = [RGBImage](repeating: RGBImage(width: 64, height: 36), count: 30)
        let empty = WindowAnalyzer(detect: { _ in [] }, embed: { _ in [] }, classifier: classifier)
        #expect(try empty.analyze(frames: frames, start: 0, end: 2).act == "not_visible")
        frames[0].pixels[0] = 255
        final class Seen { var clips: [[RGBImage]] = [] }
        let seen = Seen()
        let analyzer = WindowAnalyzer(
            detect: { _ in [Detection(score: 0.9, x0: 10, y0: 5, x1: 30, y1: 20)] },
            embed: { clip in seen.clips.append(clip); return [Float](repeating: 0, count: classifier.dimension) },
            classifier: classifier)
        let result = try analyzer.analyze(frames: frames, start: 2, end: 4)
        #expect(result.found == 4)
        #expect(result.box == squareAround(x0: 10, y0: 5, x1: 30, y1: 20, width: 64, height: 36))
        #expect(seen.clips.first?.count == 16)
        #expect(seen.clips.first?.first?.width == 256)
        #expect(classifier.classes.contains(result.act))
        #expect(abs(result.probabilities.values.reduce(0, +) - 1) < 1e-9)
    }
}
