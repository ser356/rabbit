import AppKit
import AVFoundation
import Charts
import RabbitSenseKit
import SwiftUI

// Nota: sin @State ni otras macros de SwiftUI; las Command Line Tools no traen su plugin.

// MARK: - Actos

struct ActStyle {
    let name: String
    let detail: String
    let symbol: String
    let color: Color
}

let actStyles: [String: ActStyle] = [
    "lying": ActStyle(name: "Tumbado", detail: "Descansando", symbol: "bed.double.fill", color: .blue),
    "other": ActStyle(name: "Quieto o explorando", detail: "Agazapado, olfateando o acicalándose",
                      symbol: "pawprint.fill", color: .cyan),
    "locomotion": ActStyle(name: "Moviéndose", detail: "Saltando o caminando", symbol: "hare.fill", color: .orange),
    "eating": ActStyle(name: "Comiendo", detail: "No fiable todavía", symbol: "leaf.fill", color: .purple),
    "not_visible": ActStyle(name: "No se ve", detail: "Fuera de cámara o tapado", symbol: "eye.slash",
                            color: .white.opacity(0.3)),
    "no_camera": ActStyle(name: "Sin cámara", detail: "No llegan imágenes", symbol: "video.slash",
                          color: .white.opacity(0.1)),
    "skipped": ActStyle(name: "Sin analizar", detail: "El análisis iba con retraso", symbol: "clock", color: .clear),
]

/// Actos con Cinna delante de la cámara (lo que se reparte en «Hoy»).
let cameraActs = ["lying", "other", "locomotion", "eating", "not_visible"]

func style(_ act: String) -> ActStyle {
    actStyles[act] ?? ActStyle(name: act, detail: "", symbol: "questionmark", color: .gray)
}

func duration(_ seconds: Double) -> String {
    if seconds < 60 { return seconds > 0 ? "< 1 min" : "0 min" }
    let minutes = Int((seconds / 60).rounded())
    return minutes >= 60 ? String(format: "%d h %02d min", minutes / 60, minutes % 60) : "\(minutes) min"
}

/// Inicio de la racha actual del mismo acto (sin contar las ventanas sin analizar).
func streakStart(_ records: [WindowRecord], act: String) -> Date? {
    var start: Date?
    for record in records.reversed() where record.act != "skipped" {
        if record.act != act { break }
        start = record.start
    }
    return start
}

// MARK: - App

@main
struct RabbitSenseMain: App {
    @StateObject private var monitor = Monitor()

    var body: some Scene {
        Window("RabbitSense", id: "main") {
            ContentView()
                .environmentObject(monitor)
                .frame(minWidth: 980, minHeight: 620)
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
    case days = "Días"
    var id: String { rawValue }
}

let panelWidth: CGFloat = 320

struct ContentView: View {
    @EnvironmentObject var monitor: Monitor
    @AppStorage("pagina") private var page = Page.today

    var body: some View {
        ZStack {
            Backdrop(tint: .green)
            if monitor.running {
                LiveLayer()
            } else {
                Hero()
                    .padding(.top, 56)
                    .padding(.trailing, panelWidth + 24)
            }
        }
        // Franja superior para arrastrar la ventana (no hay barra de título).
        .overlay(alignment: .top) {
            Color.clear.frame(height: 52).contentShape(Rectangle()).draggableWindow()
        }
        .overlay(alignment: .top) {
            TopBar(page: $page)
                .padding(.top, 10)
                .padding(.leading, 86)
                .padding(.trailing, 14)
        }
        .overlay(alignment: .topTrailing) {
            SidePanel(page: page)
                .padding(.top, 58)
                .padding(.trailing, 12)
                .padding(.bottom, 12)
        }
        .overlay(alignment: .top) {
            VStack(spacing: 8) {
                ForEach(monitor.alerts, id: \.identifier) { AlertBanner(alert: $0) }
            }
            .glassGroup()
            .frame(maxWidth: 460)
            .padding(.top, 58)
            .padding(.trailing, panelWidth + 24)
        }
        .overlay(alignment: .bottomLeading) {
            if monitor.running, let latest = monitor.latest {
                ActChip(latest: latest, records: monitor.today).padding(20)
            }
        }
        .ignoresSafeArea()
        .preferredColorScheme(.dark)
        .animation(.smooth(duration: 0.35), value: monitor.running)
    }
}

// MARK: - Barra superior

struct TopBar: View {
    @EnvironmentObject var monitor: Monitor
    @Binding var page: Page

    var body: some View {
        ZStack {
            PagePicker(page: $page)
            HStack {
                StatusCapsule()
                Spacer()
                if monitor.running {
                    GlassButton(title: "Pausar", symbol: "pause.fill") { monitor.stop() }
                }
            }
        }
        .glassGroup(spacing: 16)
    }
}

struct StatusCapsule: View {
    @EnvironmentObject var monitor: Monitor

    var body: some View {
        HStack(spacing: 7) {
            Circle()
                .fill(monitor.running ? Color.green : monitor.ready ? Color.gray : Color.yellow)
                .frame(width: 8, height: 8)
            Text(monitor.running ? "Observando" : monitor.ready ? "En pausa" : "Preparando")
                .fontWeight(.semibold)
            if monitor.running {
                Text(monitor.cameraName).foregroundStyle(.secondary)
            }
        }
        .font(.callout)
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .glassSurface(Capsule())
    }
}

/// Selector de página: cápsula de cristal con la opción elegida resaltada.
struct PagePicker: View {
    @Binding var page: Page
    @Namespace private var selection

    var body: some View {
        HStack(spacing: 2) {
            ForEach(Page.allCases) { option in
                Button {
                    withAnimation(.smooth(duration: 0.3)) { page = option }
                } label: {
                    Text(option.rawValue)
                        .font(.callout.weight(.semibold))
                        .foregroundStyle(page == option ? Color.white : Color.white.opacity(0.6))
                        .padding(.horizontal, 18)
                        .padding(.vertical, 6)
                        .background {
                            if page == option {
                                Capsule().fill(Color.white.opacity(0.18))
                                    .matchedGeometryEffect(id: "elegida", in: selection)
                            }
                        }
                        .contentShape(Capsule())
                }
                .buttonStyle(.plain)
            }
        }
        .padding(3)
        .glassSurface(Capsule(), interactive: true)
    }
}

/// Botón de cristal dibujado por la app (los de la barra del sistema no respetan el estilo).
struct GlassButton: View {
    enum Size { case small, regular, large }

    let title: String
    let symbol: String
    var tint: Color?
    var size = Size.regular
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Label(title, systemImage: symbol)
                .font(size == .large ? .title3.weight(.semibold) : size == .small ? .callout.weight(.medium)
                      : .body.weight(.semibold))
                .foregroundStyle(.white)
                .padding(.horizontal, size == .large ? 28 : size == .small ? 14 : 16)
                .padding(.vertical, size == .large ? 14 : size == .small ? 6 : 8)
                .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .glassSurface(Capsule(), tint: tint, interactive: true)
    }
}

// MARK: - Contenido principal

/// En pausa: quién, qué falta y el botón para empezar.
struct Hero: View {
    @EnvironmentObject var monitor: Monitor

    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: monitor.problem == nil ? "hare.fill" : "exclamationmark.triangle.fill")
                .font(.system(size: 54, weight: .medium))
                .foregroundStyle(monitor.problem == nil ? Color.white.opacity(0.9) : Color.yellow)
                .padding(.bottom, 4)
            Text("Cinna").font(.system(size: 34, weight: .bold))
            Group {
                if let problem = monitor.problem {
                    Text(problem)
                } else if !monitor.ready {
                    Text("Preparando los modelos… La primera vez tarda un par de minutos.")
                } else {
                    Text("Deja el iPhone cerca del Mac, bloqueado y apoyado en horizontal. No se guarda vídeo: solo lo que hace.")
                }
            }
            .font(.body)
            .foregroundStyle(.secondary)
            .multilineTextAlignment(.center)
            .frame(maxWidth: 380)

            if monitor.ready {
                GlassButton(title: monitor.problem == nil ? "Empezar a observar" : "Volver a intentarlo",
                            symbol: "play.fill", tint: .green, size: .large) { monitor.start() }
                    .padding(.top, 14)
            } else {
                ProgressView().controlSize(.regular).padding(.top, 14)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

/// Observando: la cámara a pantalla completa y el recuadro del seguidor.
struct LiveLayer: View {
    @EnvironmentObject var monitor: Monitor

    var body: some View {
        GeometryReader { geo in
            let video = fillRect(size: geo.size, aspect: monitor.frameAspect)
            ZStack(alignment: .topLeading) {
                CameraPreview(session: monitor.session)
                if let box = monitor.box {
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .stroke(style(monitor.latest?.act ?? "other").color, lineWidth: 2.5)
                        .shadow(color: .black.opacity(0.4), radius: 3)
                        .frame(width: box.width * video.width, height: box.height * video.height)
                        .position(x: video.minX + box.midX * video.width, y: video.minY + box.midY * video.height)
                        .animation(.linear(duration: 0.08), value: box)
                }
            }
        }
        .background(Color.black)
    }
}

/// Rectángulo que ocupa un vídeo de proporción `aspect` al llenar `size` (recortando lo que sobra).
func fillRect(size: CGSize, aspect: CGFloat) -> CGRect {
    guard size.width > 0, size.height > 0, aspect > 0 else { return .zero }
    if size.width / size.height > aspect {
        let height = size.width / aspect
        return CGRect(x: 0, y: (size.height - height) / 2, width: size.width, height: height)
    }
    let width = size.height * aspect
    return CGRect(x: (size.width - width) / 2, y: 0, width: width, height: size.height)
}

struct ActChip: View {
    let latest: WindowRecord
    let records: [WindowRecord]

    var body: some View {
        let s = style(latest.act)
        HStack(spacing: 12) {
            Image(systemName: s.symbol)
                .font(.system(size: 17, weight: .semibold))
                .foregroundStyle(.white)
                .frame(width: 38, height: 38)
                .background(Circle().fill(s.color))
            VStack(alignment: .leading, spacing: 1) {
                Text(s.name).font(.title3.weight(.semibold))
                if let since = streakStart(records, act: latest.act) {
                    Text("desde las \(since.formatted(date: .omitted, time: .shortened)) · \(duration(latest.end.timeIntervalSince(since)))")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .padding(.leading, 8)
        .padding(.trailing, 20)
        .padding(.vertical, 8)
        .glassSurface(Capsule())
        .animation(.smooth, value: latest.act)
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
        .padding(14)
        .glassSurface(RoundedRectangle(cornerRadius: 20, style: .continuous), tint: .orange.opacity(0.35))
    }
}

/// Imagen en directo de la cámara, llenando su espacio.
struct CameraPreview: NSViewRepresentable {
    let session: AVCaptureSession

    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        let layer = AVCaptureVideoPreviewLayer(session: session)
        layer.videoGravity = .resizeAspectFill
        layer.backgroundColor = NSColor.black.cgColor
        view.layer = layer
        view.wantsLayer = true
        return view
    }

    func updateNSView(_ nsView: NSView, context: Context) {}
}

// MARK: - Panel lateral

/// Panel de cristal que se ajusta a su contenido y solo se desplaza si no cabe.
struct SidePanel: View {
    let page: Page

    var body: some View {
        ViewThatFits(in: .vertical) {
            content
            ScrollView { content }.scrollIndicators(.never)
        }
        .frame(width: panelWidth)
        .clipShape(RoundedRectangle(cornerRadius: 24, style: .continuous))
        .glassSurface(RoundedRectangle(cornerRadius: 24, style: .continuous))
        .frame(maxHeight: .infinity, alignment: .top)
    }

    var content: some View {
        VStack(alignment: .leading, spacing: 24) {
            switch page {
            case .today: TodayPanel()
            case .days: DaysPanel()
            }
            Text("Comer aún no es fiable y el movimiento está sin validar. No es un diagnóstico.")
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
        .padding(20)
        .frame(width: panelWidth, alignment: .leading)
    }
}

/// Sección del panel: título pequeño y contenido, sin fondo propio (no se pone cristal sobre cristal).
struct PanelSection<Content: View>: View {
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
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

struct BigNumber: View {
    let value: String
    let unit: String

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(value).font(.system(size: 32, weight: .bold, design: .rounded)).monospacedDigit()
            Text(unit).font(.callout).foregroundStyle(.secondary)
        }
    }
}

struct TodayPanel: View {
    @EnvironmentObject var monitor: Monitor

    var body: some View {
        let summary = ActivitySummary(monitor.today)
        let withCamera = cameraActs.reduce(0) { $0 + (summary.secondsByAct[$1] ?? 0) }
        let present = cameraActs.filter { (summary.secondsByAct[$0] ?? 0) > 0 }
        let noCamera = summary.secondsByAct["no_camera"] ?? 0

        PanelSection("Hoy") {
            if summary.visibleSeconds > 0 {
                BigNumber(value: duration(summary.visibleSeconds), unit: "viéndole")
                Text("de \(duration(withCamera)) con cámara").font(.caption).foregroundStyle(.secondary)
                StackedBar(parts: present.map { (style($0).color, summary.secondsByAct[$0] ?? 0) })
                    .padding(.vertical, 4)
                ForEach(present, id: \.self) { act in
                    let seconds = summary.secondsByAct[act] ?? 0
                    ActRow(act: act, seconds: seconds, share: withCamera > 0 ? seconds / withCamera : 0)
                }
            } else {
                Text("Aún no le has visto hoy. Cuando empieces, aquí verás cuánto tiempo pasa tumbado, quieto o moviéndose.")
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }

        if !monitor.today.isEmpty {
            PanelSection("Día") {
                DayTimeline(records: monitor.today).frame(height: 18)
                HourLabels()
                if noCamera >= 60 {
                    Text("\(duration(noCamera)) sin imagen").font(.caption).foregroundStyle(.secondary)
                }
            }
        }

        if let rate = summary.displacementPerMinute {
            PanelSection("Movimiento") {
                BigNumber(value: String(format: "%.2f", rate), unit: "cuerpos por minuto")
            }
        }
    }
}

struct ActRow: View {
    let act: String
    let seconds: Double
    let share: Double

    var body: some View {
        let s = style(act)
        HStack(spacing: 9) {
            Circle().fill(s.color).frame(width: 9, height: 9)
            Text(s.name)
            if act == "eating" { Text("no fiable").font(.caption).foregroundStyle(.tertiary) }
            Spacer()
            Text(duration(seconds)).foregroundStyle(.secondary).monospacedDigit()
            Text("\(Int((100 * share).rounded())) %")
                .foregroundStyle(.tertiary)
                .monospacedDigit()
                .frame(width: 42, alignment: .trailing)
        }
        .font(.callout)
    }
}

/// Una sola barra con el reparto del tiempo.
struct StackedBar: View {
    let parts: [(Color, Double)]

    var body: some View {
        GeometryReader { geo in
            let total = parts.reduce(0) { $0 + $1.1 }
            let gaps = CGFloat(max(0, parts.count - 1)) * 2
            HStack(spacing: 2) {
                ForEach(parts.indices, id: \.self) { i in
                    Rectangle().fill(parts[i].0)
                        .frame(width: total > 0 ? (geo.size.width - gaps) * parts[i].1 / total : 0)
                }
            }
        }
        .frame(height: 10)
        .clipShape(Capsule())
    }
}

/// Las 24 h del día: un color por acto, el tiempo sin cámara en gris y una marca en la hora actual.
struct DayTimeline: View {
    let records: [WindowRecord]

    var body: some View {
        TimelineView(.periodic(from: .now, by: 60)) { context in
            Canvas { canvas, size in
                let day = Calendar.current.startOfDay(for: context.date)
                canvas.fill(Path(CGRect(origin: .zero, size: size)), with: .color(.white.opacity(0.07)))
                for record in records where record.act != "skipped" {
                    let x = record.start.timeIntervalSince(day) / 86400 * size.width
                    let width = max(1, record.seconds / 86400 * size.width)
                    canvas.fill(Path(CGRect(x: x, y: 0, width: width, height: size.height)),
                                with: .color(style(record.act).color))
                }
                let now = context.date.timeIntervalSince(day) / 86400 * size.width
                canvas.fill(Path(CGRect(x: now - 1, y: 0, width: 2, height: size.height)), with: .color(.white))
            }
            .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
        }
    }
}

struct HourLabels: View {
    var body: some View {
        HStack {
            ForEach(["0", "6", "12", "18", "24"], id: \.self) { hour in
                Text(hour)
                if hour != "24" { Spacer() }
            }
        }
        .font(.caption2)
        .foregroundStyle(.tertiary)
    }
}

// MARK: - Días

struct DayRow: Identifiable {
    let id = UUID()
    let day: Date
    let name: String
    let hours: Double
}

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
            return cameraActs.compactMap { act in
                let seconds = summary.secondsByAct[act] ?? 0
                return seconds > 0 ? DayRow(day: day, name: style(act).name, hours: seconds / 3600) : nil
            }
        }
        let count = monitor.baselineDays

        PanelSection("Línea base") {
            BigNumber(value: "\(min(count, 7)) / 7", unit: "días")
            ProgressView(value: Double(min(count, 7)), total: 7).tint(.green)
            Text(count >= 7
                 ? "Ya compara cada tramo del día con los mismos tramos de otros días."
                 : "Cuenta los días en que se le ve al menos 1 h. Con 7, avisa si se mueve mucho menos de lo normal.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            GlassButton(title: "Empezar de cero…", symbol: "arrow.counterclockwise", size: .small) {
                monitor.resetJournal()
            }
            .help("Borra el diario (va a la Papelera) y la línea base vuelve a 0")
            .padding(.top, 2)
        }

        if !rows.isEmpty {
            PanelSection("Horas por día") {
                Chart(rows) { row in
                    BarMark(x: .value("Día", row.day, unit: .day), y: .value("Horas", row.hours))
                        .foregroundStyle(by: .value("Acto", row.name))
                        .cornerRadius(3)
                }
                .chartForegroundStyleScale(domain: cameraActs.map { style($0).name },
                                           range: cameraActs.map { style($0).color })
                .chartXAxis {
                    AxisMarks(values: .stride(by: .day)) { _ in
                        AxisValueLabel(format: .dateTime.day().month(.abbreviated), centered: true)
                    }
                }
                .chartLegend(position: .bottom, alignment: .leading)
                .frame(height: 200)
            }
        }

        if !days.isEmpty {
            PanelSection("Resumen") {
                ForEach(days.indices, id: \.self) { i in
                    let summary = ActivitySummary(days[i])
                    HStack {
                        Text(days[i][0].start.formatted(.dateTime.weekday(.abbreviated).day().month(.abbreviated)))
                        Spacer()
                        Text(duration(summary.visibleSeconds)).foregroundStyle(.secondary).monospacedDigit()
                        if let rate = summary.displacementPerMinute {
                            Text(String(format: "%.2f/min", rate))
                                .foregroundStyle(.tertiary)
                                .monospacedDigit()
                                .frame(width: 70, alignment: .trailing)
                        }
                    }
                    .font(.callout)
                }
            }
        }
    }
}
