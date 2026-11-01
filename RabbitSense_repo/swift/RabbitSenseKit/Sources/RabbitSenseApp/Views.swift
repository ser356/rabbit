import AppKit
import AVFoundation
import Charts
import RabbitSenseKit
import SwiftUI

// MARK: - Actos

struct ActStyle {
    let name: String
    let detail: String
    let symbol: String
    let color: Color
}

let actStyles: [String: ActStyle] = [
    "lying": ActStyle(name: "Tumbado", detail: "Descansando estirado", symbol: "bed.double.fill", color: .blue),
    "other": ActStyle(name: "Quieto o explorando", detail: "Agazapado, olfateando o acicalándose",
                      symbol: "pawprint.fill", color: .teal),
    "locomotion": ActStyle(name: "Moviéndose", detail: "Saltando o caminando", symbol: "hare.fill", color: .orange),
    "eating": ActStyle(name: "Comiendo", detail: "No fiable todavía", symbol: "leaf.fill", color: .purple),
    "not_visible": ActStyle(name: "No se ve", detail: "Fuera de cámara o tapado", symbol: "eye.slash", color: .gray),
    "no_camera": ActStyle(name: "Sin cámara", detail: "No llegan imágenes", symbol: "video.slash", color: .red),
    "skipped": ActStyle(name: "Sin analizar", detail: "El análisis iba con retraso", symbol: "clock",
                        color: .gray.opacity(0.35)),
]

/// Orden en que se enseñan los actos.
let shownActs = ["lying", "other", "locomotion", "eating", "not_visible", "no_camera"]

func style(_ act: String) -> ActStyle {
    actStyles[act] ?? ActStyle(name: act, detail: "", symbol: "questionmark", color: .gray)
}

func duration(_ seconds: Double) -> String {
    let minutes = Int((seconds / 60).rounded())
    if seconds < 60 { return seconds > 0 ? "< 1 min" : "0 min" }
    return minutes >= 60 ? String(format: "%d h %02d min", minutes / 60, minutes % 60) : "\(minutes) min"
}

// MARK: - App

@main
struct RabbitSenseMain: App {
    @StateObject private var monitor = Monitor()

    var body: some Scene {
        Window("RabbitSense", id: "main") {
            ContentView()
                .environmentObject(monitor)
                .frame(minWidth: 1040, minHeight: 640)
        }
        .windowStyle(.hiddenTitleBar)

        MenuBarExtra {
            MenuContent().environmentObject(monitor)
        } label: {
            Image(systemName: monitor.running ? "hare.fill" : "hare")
        }
    }
}

/// Menú de la barra superior: la app sigue observando aunque se cierre la ventana.
struct MenuContent: View {
    @EnvironmentObject var monitor: Monitor
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        Text(monitor.running ? "Observando" : "En pausa")
        if let latest = monitor.latest, monitor.running {
            Text("Ahora: \(style(latest.act).name)")
        }
        Divider()
        Button(monitor.running ? "Pausar" : "Empezar") { monitor.toggle() }
            .disabled(!monitor.ready)
        Button("Abrir RabbitSense") {
            openWindow(id: "main")
            NSApp.activate(ignoringOtherApps: true)
        }
        Button("Empezar el diario de cero…") { monitor.resetJournal() }
        Divider()
        Button("Salir") { NSApp.terminate(nil) }
    }
}

// MARK: - Ventana

enum Page: String, CaseIterable, Identifiable {
    case today = "Hoy"
    case days = "Últimos días"
    var id: String { rawValue }
}

struct ContentView: View {
    @EnvironmentObject var monitor: Monitor
    // @AppStorage es un envoltorio normal; @State es una macro y las Command Line Tools no traen su plugin.
    @AppStorage("pagina") private var page = Page.today

    var body: some View {
        HStack(alignment: .top, spacing: 20) {
            VStack(alignment: .leading, spacing: 12) {
                LiveView()
                ForEach(monitor.alerts, id: \.identifier) { AlertBanner(alert: $0) }
                Spacer(minLength: 0)
            }
            .frame(minWidth: 560, maxWidth: .infinity)

            ScrollView {
                VStack(spacing: 14) {
                    switch page {
                    case .today: TodayPanel()
                    case .days: DaysPanel()
                    }
                }
                .padding(.bottom, 8)
            }
            .scrollIndicators(.never)
            .frame(width: 360)
        }
        .padding(.horizontal, 20)
        .padding(.bottom, 20)
        .safeAreaInset(edge: .top, spacing: 0) {
            // Barra de arriba con controles de cristal (la ventana no tiene barra de título).
            ZStack {
                HStack(spacing: 12) {
                    StatusPill()
                    Label("RabbitSense", systemImage: "hare")
                        .font(.title3.weight(.semibold))
                    Spacer()
                }
                HStack(spacing: 10) {
                    PagePicker(page: $page)
                    GlassButton(title: monitor.running ? "Pausar" : "Empezar",
                                symbol: monitor.running ? "pause.fill" : "play.fill",
                                tint: monitor.running ? .orange : .green) { monitor.toggle() }
                        .disabled(!monitor.ready)
                        .opacity(monitor.ready ? 1 : 0.5)
                        .help(monitor.running ? "Dejar de observar" : "Empezar a observar con la cámara")
                }
                .glassGroup(spacing: 10)
            }
            .padding(.leading, 86)
            .padding(.trailing, 16)
            .padding(.top, 10)
            .padding(.bottom, 16)
            .background(Color.clear.contentShape(Rectangle()).draggableWindow())
        }
        .ignoresSafeArea(edges: .top)
    }
}

struct StatusPill: View {
    @EnvironmentObject var monitor: Monitor

    var body: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(monitor.running ? Color.green : monitor.ready ? Color.gray : Color.yellow)
                .frame(width: 8, height: 8)
            Text(monitor.running ? "Observando" : monitor.ready ? "En pausa" : "Preparando")
                .fontWeight(.medium)
            if monitor.running {
                Text("· \(monitor.cameraName)").foregroundStyle(.secondary)
            }
        }
        .font(.callout)
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .glassSurface(Capsule())
    }
}

// MARK: - Imagen en directo

struct LiveView: View {
    @EnvironmentObject var monitor: Monitor

    var body: some View {
        ZStack(alignment: .bottomLeading) {
            CameraPreview(session: monitor.session)
            if monitor.running, let box = monitor.box {
                GeometryReader { geo in
                    let rect = CGRect(x: box.minX * geo.size.width, y: box.minY * geo.size.height,
                                      width: box.width * geo.size.width, height: box.height * geo.size.height)
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(style(monitor.latest?.act ?? "other").color, lineWidth: 2.5)
                        .shadow(color: .black.opacity(0.5), radius: 3)
                        .frame(width: rect.width, height: rect.height)
                        .position(x: rect.midX, y: rect.midY)
                        .animation(.linear(duration: 0.08), value: box)
                }
            }
            if !monitor.running {
                Placeholder()
            } else if let latest = monitor.latest {
                ActChip(record: latest).padding(14)
            }
        }
        .aspectRatio(monitor.frameAspect, contentMode: .fit)
        .background(Color.black)
        .clipShape(RoundedRectangle(cornerRadius: 14))
        .overlay(RoundedRectangle(cornerRadius: 14).strokeBorder(Color.primary.opacity(0.08)))
        .environment(\.colorScheme, .dark)
    }
}

struct Placeholder: View {
    @EnvironmentObject var monitor: Monitor

    var body: some View {
        VStack(spacing: 12) {
            if let problem = monitor.problem {
                Image(systemName: "exclamationmark.triangle.fill").font(.system(size: 38)).foregroundStyle(.yellow)
                Text("No puedo observar").font(.title2.weight(.semibold))
                Text(problem).foregroundStyle(.secondary)
            } else if !monitor.ready {
                ProgressView().controlSize(.large)
                Text("Preparando los modelos…").font(.title2.weight(.semibold))
                Text("La primera vez tarda un par de minutos.").foregroundStyle(.secondary)
            } else {
                Image(systemName: "hare").font(.system(size: 44, weight: .light))
                Text("Listo para observar a Cinna").font(.title2.weight(.semibold))
                Text("Deja el iPhone cerca del Mac, bloqueado, apoyado y en horizontal, y pulsa Empezar. No se guarda vídeo.")
                    .foregroundStyle(.secondary)
            }
        }
        .multilineTextAlignment(.center)
        .frame(maxWidth: 420)
        .foregroundStyle(.white)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

struct ActChip: View {
    let record: WindowRecord

    var body: some View {
        let s = style(record.act)
        HStack(spacing: 8) {
            Image(systemName: s.symbol).foregroundStyle(s.color)
            Text(s.name).fontWeight(.semibold)
            if let confidence = record.confidence {
                Text("\(Int((100 * confidence).rounded())) %").foregroundStyle(.secondary).monospacedDigit()
            }
        }
        .font(.title3)
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .background(.ultraThinMaterial, in: Capsule())
    }
}

struct AlertBanner: View {
    let alert: RabbitAlert

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange).font(.title3)
            VStack(alignment: .leading, spacing: 2) {
                Text(alert.title).fontWeight(.semibold)
                Text(alert.body).foregroundStyle(.secondary)
            }
            Spacer(minLength: 0)
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 10).fill(Color.orange.opacity(0.12)))
    }
}

/// Imagen en directo de la cámara.
struct CameraPreview: NSViewRepresentable {
    let session: AVCaptureSession

    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        let layer = AVCaptureVideoPreviewLayer(session: session)
        layer.videoGravity = .resizeAspect
        layer.backgroundColor = NSColor.black.cgColor
        view.layer = layer
        view.wantsLayer = true
        return view
    }

    func updateNSView(_ nsView: NSView, context: Context) {}
}

// MARK: - Panel de hoy

struct Card<Content: View>: View {
    let title: String
    let content: Content

    init(_ title: String, @ViewBuilder content: () -> Content) {
        self.title = title
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title.uppercased())
                .font(.caption.weight(.semibold))
                .tracking(0.6)
                .foregroundStyle(.secondary)
            content
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 12).fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(Color.primary.opacity(0.07)))
    }
}

struct TodayPanel: View {
    @EnvironmentObject var monitor: Monitor

    var body: some View {
        let summary = ActivitySummary(monitor.today)
        let total = shownActs.reduce(0) { $0 + (summary.secondsByAct[$1] ?? 0) }

        Card("Ahora") { NowView(records: monitor.today, latest: monitor.latest) }

        Card("Hoy") {
            VStack(spacing: 4) {
                Timeline(records: monitor.today).frame(height: 34)
                HStack {
                    ForEach(["0", "6", "12", "18", "24"], id: \.self) { hour in
                        Text(hour)
                        if hour != "24" { Spacer() }
                    }
                }
                .font(.caption2)
                .foregroundStyle(.secondary)
            }
            Divider().padding(.vertical, 2)
            ForEach(shownActs, id: \.self) { act in
                let seconds = summary.secondsByAct[act] ?? 0
                ActRow(act: act, seconds: seconds, share: total > 0 ? seconds / total : 0)
            }
        }

        Card("Movimiento") {
            if let rate = summary.displacementPerMinute {
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Text(String(format: "%.2f", rate))
                        .font(.system(size: 30, weight: .semibold, design: .rounded))
                        .monospacedDigit()
                    Text("cuerpos por minuto").foregroundStyle(.secondary)
                }
                Text("Cuánto se desplaza, medido en tamaños de su propio cuerpo.")
                    .font(.caption).foregroundStyle(.secondary)
            } else {
                Text("Todavía sin datos de hoy.").foregroundStyle(.secondary)
            }
        }

        Text("Comer no es fiable todavía y el movimiento con cámara fija está sin validar. No es un diagnóstico.")
            .font(.caption)
            .foregroundStyle(.tertiary)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 4)
    }
}

struct NowView: View {
    let records: [WindowRecord]
    let latest: WindowRecord?

    var body: some View {
        if let latest {
            let s = style(latest.act)
            HStack(spacing: 14) {
                Image(systemName: s.symbol)
                    .font(.system(size: 22, weight: .semibold))
                    .foregroundStyle(.white)
                    .frame(width: 48, height: 48)
                    .background(Circle().fill(s.color))
                VStack(alignment: .leading, spacing: 3) {
                    Text(s.name).font(.title3.weight(.semibold))
                    Text(s.detail).font(.callout).foregroundStyle(.secondary)
                    if let since = streakStart(latest.act) {
                        Text("Desde las \(since.formatted(date: .omitted, time: .shortened)) · \(duration(latest.end.timeIntervalSince(since)))")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
                Spacer(minLength: 0)
            }
        } else {
            Text("Aún no hay ninguna observación hoy.").foregroundStyle(.secondary)
        }
    }

    /// Inicio de la racha actual del mismo acto (sin contar las ventanas sin analizar).
    func streakStart(_ act: String) -> Date? {
        var start: Date?
        for record in records.reversed() where record.act != "skipped" {
            if record.act != act { break }
            start = record.start
        }
        return start
    }
}

struct ActRow: View {
    let act: String
    let seconds: Double
    let share: Double

    var body: some View {
        let s = style(act)
        HStack(spacing: 10) {
            Image(systemName: s.symbol).foregroundStyle(s.color).frame(width: 20)
            VStack(alignment: .leading, spacing: 5) {
                HStack(alignment: .firstTextBaseline) {
                    Text(s.name)
                    if act == "eating" { Text("no fiable").font(.caption).foregroundStyle(.secondary) }
                    Spacer()
                    Text(duration(seconds)).monospacedDigit().foregroundStyle(.secondary)
                }
                ShareBar(value: share, color: s.color)
            }
        }
    }
}

struct ShareBar: View {
    let value: Double
    let color: Color

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                Capsule().fill(Color.primary.opacity(0.07))
                Capsule().fill(color).frame(width: max(0, min(1, value)) * geo.size.width)
            }
        }
        .frame(height: 4)
    }
}

/// Las 24 h del día, con un color por acto y una marca en la hora actual.
struct Timeline: View {
    let records: [WindowRecord]

    var body: some View {
        TimelineView(.periodic(from: .now, by: 60)) { context in
            Canvas { canvas, size in
                let day = Calendar.current.startOfDay(for: context.date)
                canvas.fill(Path(roundedRect: CGRect(origin: .zero, size: size), cornerRadius: 6),
                            with: .color(.primary.opacity(0.06)))
                for hour in stride(from: 6.0, to: 24, by: 6) {
                    let x = hour / 24 * size.width
                    canvas.fill(Path(CGRect(x: x, y: 0, width: 1, height: size.height)), with: .color(.primary.opacity(0.08)))
                }
                for record in records {
                    let x = record.start.timeIntervalSince(day) / 86400 * size.width
                    let width = max(1, record.seconds / 86400 * size.width)
                    canvas.fill(Path(CGRect(x: x, y: 0, width: width, height: size.height)),
                                with: .color(style(record.act).color))
                }
                let now = context.date.timeIntervalSince(day) / 86400 * size.width
                canvas.fill(Path(CGRect(x: now - 1, y: -2, width: 2, height: size.height + 4)), with: .color(.primary))
            }
            .clipShape(RoundedRectangle(cornerRadius: 6))
        }
    }
}

// MARK: - Últimos días

struct DayRow: Identifiable {
    let id = UUID()
    let day: Date
    let name: String
    let hours: Double
}

/// La gráfica solo enseña lo que se vio (el tiempo sin cámara va en el resumen).
let chartActs = ["lying", "other", "locomotion", "eating", "not_visible"]

struct DaysPanel: View {
    @EnvironmentObject var monitor: Monitor

    var days: [[WindowRecord]] {
        (monitor.today.isEmpty ? [] : [monitor.today]) + monitor.pastDays
    }

    var body: some View {
        let rows = days.flatMap { records -> [DayRow] in
            guard let first = records.first else { return [] }
            let day = Calendar.current.startOfDay(for: first.start)
            let summary = ActivitySummary(records)
            return chartActs.compactMap { act in
                let seconds = summary.secondsByAct[act] ?? 0
                return seconds > 0 ? DayRow(day: day, name: style(act).name, hours: seconds / 3600) : nil
            }
        }

        Card("Horas por día") {
            if rows.isEmpty {
                Text("Aún no hay días con diario.").foregroundStyle(.secondary)
            } else {
                Chart(rows) { row in
                    BarMark(x: .value("Día", row.day, unit: .day), y: .value("Horas", row.hours))
                        .foregroundStyle(by: .value("Acto", row.name))
                }
                .chartForegroundStyleScale(domain: chartActs.map { style($0).name },
                                           range: chartActs.map { style($0).color })
                .chartXAxis {
                    AxisMarks(values: .stride(by: .day)) { _ in
                        AxisGridLine()
                        AxisValueLabel(format: .dateTime.day().month(.abbreviated), centered: true)
                    }
                }
                .chartYAxisLabel("horas")
                .chartLegend(position: .bottom, alignment: .leading)
                .frame(height: 230)
            }
        }

        Card("Línea base") {
            let count = monitor.baselineDays
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Text("\(min(count, 7)) / 7")
                    .font(.system(size: 26, weight: .semibold, design: .rounded))
                    .monospacedDigit()
                Text("días previos").foregroundStyle(.secondary)
            }
            ProgressView(value: Double(min(count, 7)), total: 7)
            Text(count >= 7
                 ? "Ya compara la actividad de hoy con la de los mismos tramos de otros días."
                 : "Con 7 días en los que se le vea al menos 1 h, empieza a avisar si se mueve mucho menos de lo normal.")
                .font(.caption).foregroundStyle(.secondary)
            GlassButton(title: "Empezar de cero…", symbol: "arrow.counterclockwise", size: .small) {
                monitor.resetJournal()
            }
            .help("Borra el diario (va a la Papelera) y la línea base vuelve a 0")
        }

        if !days.isEmpty {
            Card("Resumen") {
                ForEach(days.indices, id: \.self) { i in
                    let summary = ActivitySummary(days[i])
                    HStack {
                        Text(days[i][0].start.formatted(.dateTime.weekday(.abbreviated).day().month(.abbreviated)))
                        Spacer()
                        Text("visible \(duration(summary.visibleSeconds))").foregroundStyle(.secondary)
                        if let noCamera = summary.secondsByAct["no_camera"], noCamera >= 60 {
                            Text("· sin cámara \(duration(noCamera))").foregroundStyle(.tertiary)
                        }
                        if let rate = summary.displacementPerMinute {
                            Text(String(format: "%.2f/min", rate)).monospacedDigit().frame(width: 72, alignment: .trailing)
                        }
                    }
                    .font(.callout)
                    if i < days.count - 1 { Divider() }
                }
            }
        }
    }
}
