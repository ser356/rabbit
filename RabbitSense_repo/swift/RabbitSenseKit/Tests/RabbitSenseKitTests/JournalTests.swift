import Foundation
import Testing
@testable import RabbitSenseKit

/// Ventanas de 2 s seguidas desde `start`, con el acto y el desplazamiento dados.
func windows(from start: Date, count: Int, act: String, displacement: Double? = nil) -> [WindowRecord] {
    (0..<count).map { k in
        WindowRecord(start: start.addingTimeInterval(Double(2 * k)), end: start.addingTimeInterval(Double(2 * k + 2)),
                     act: act, confidence: 0.8, displacement: displacement)
    }
}

struct JournalTests {
    let calendar: Calendar = {
        var c = Calendar(identifier: .gregorian)
        c.timeZone = TimeZone(identifier: "Europe/Madrid")!
        return c
    }()

    func date(_ day: Int, _ hour: Int, _ minute: Int = 0) -> Date {
        calendar.date(from: DateComponents(year: 2026, month: 11, day: day, hour: hour, minute: minute))!
    }

    @Test func summaryCountsVisibleTimeAndDisplacement() {
        let records = windows(from: date(1, 10), count: 30, act: "lying", displacement: 0) +
            windows(from: date(1, 10, 1), count: 15, act: "locomotion", displacement: 1) +
            windows(from: date(1, 10, 2), count: 10, act: "not_visible")
        let s = ActivitySummary(records)
        #expect(s.visibleSeconds == 90)
        #expect(s.share("lying") == 60.0 / 90)
        #expect(s.displacement == 15)
        #expect(s.displacementPerMinute == 15 / 1.5)
    }

    @Test func journalRoundTripsByDay() throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let journal = Journal(folder: folder, calendar: calendar)
        let first = windows(from: date(2, 9), count: 3, act: "other", displacement: 0.2)
        let next = windows(from: date(3, 9), count: 2, act: "lying")
        for r in first + next { try journal.append(r) }
        #expect(journal.records(on: date(2, 12)) == first)
        #expect(journal.records(on: date(3, 23)) == next)
        #expect(journal.history(before: date(4, 8), days: 3).map(\.count) == [2, 3])
        try? FileManager.default.removeItem(at: folder)
    }

    @Test func cameraAndVisibilityAlertsNeedNoBaseline() {
        var rules = AlertRules()
        rules.notVisibleMinutes = 5
        let today = windows(from: date(5, 10), count: 30, act: "other") + windows(from: date(5, 10, 1), count: 160, act: "no_camera")
        let alerts = rules.evaluate(today: today, history: [], now: date(5, 10, 7), calendar: calendar)
        #expect(alerts.map(\.identifier) == ["sin-camara"])
        let hidden = windows(from: date(5, 11), count: 160, act: "not_visible")
        #expect(rules.evaluate(today: hidden, history: [], now: date(5, 11, 6), calendar: calendar).map(\.identifier) == ["no-visible"])
    }

    @Test func lowActivityAlertComparesTheSameHoursOfPreviousDays() {
        let rules = AlertRules()
        let history = (1...7).map { d in windows(from: date(d, 9), count: 2700, act: "other", displacement: 0.5) }
        let quiet = windows(from: date(8, 9), count: 2700, act: "lying", displacement: 0.1)
        let alerts = rules.evaluate(today: quiet, history: history, now: date(8, 12), calendar: calendar)
        #expect(alerts.map(\.identifier) == ["actividad-baja"])
        let usual = windows(from: date(8, 9), count: 2700, act: "other", displacement: 0.45)
        #expect(rules.evaluate(today: usual, history: history, now: date(8, 12), calendar: calendar).isEmpty)
        #expect(rules.evaluate(today: quiet, history: Array(history.prefix(6)), now: date(8, 12), calendar: calendar).isEmpty)
    }

    @Test func trackerBoxesReplaceDetectionsInLiveMode() throws {
        let classifier = try ActClassifier.bundled()
        let analyzer = WindowAnalyzer(detect: { _ in [] },
                                      embed: { _ in [Float](repeating: 0, count: classifier.dimension) }, classifier: classifier)
        let frames = [RGBImage](repeating: RGBImage(width: 64, height: 36), count: 30)
        let lost = [Detection?](repeating: nil, count: 30)
        #expect(try analyzer.analyze(frames: frames, boxes: lost, start: 0, end: 2).act == "not_visible")
        var boxes = lost
        boxes[0] = Detection(score: 1, x0: 10, y0: 5, x1: 30, y1: 20)
        boxes[29] = Detection(score: 1, x0: 20, y0: 5, x1: 40, y1: 25)
        let result = try analyzer.analyze(frames: frames, boxes: boxes, start: 0, end: 2)
        #expect(result.found == 2)
        #expect(result.box == squareAround(x0: 10, y0: 5, x1: 40, y1: 25, width: 64, height: 36))
    }
}
