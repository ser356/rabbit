import AVFoundation
import RabbitSenseKit
import SwiftUI

let actNames = ["locomotion": "Desplazándose", "lying": "Tumbado", "other": "Agazapado, explorando o acicalándose",
                "eating": "Comiendo (no fiable)", "not_visible": "No se ve", "no_camera": "Sin cámara",
                "skipped": "Sin analizar"]
let actColors: [String: Color] = ["locomotion": .orange, "lying": .blue, "other": .green, "eating": .purple,
                                  "not_visible": .gray.opacity(0.5), "no_camera": .red.opacity(0.6),
                                  "skipped": .gray.opacity(0.2)]

@main
struct RabbitSenseMain: App {
    @StateObject private var monitor = Monitor()

    var body: some Scene {
        WindowGroup("RabbitSense") {
            ContentView().environmentObject(monitor).frame(minWidth: 900, minHeight: 600)
        }
    }
}

struct ContentView: View {
    @EnvironmentObject var monitor: Monitor

    var body: some View {
        HStack(alignment: .top, spacing: 16) {
            VStack(alignment: .leading, spacing: 12) {
                CameraPreview(session: monitor.session)
                    .aspectRatio(16 / 9, contentMode: .fit)
                    .background(Color.black)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                HStack {
                    Button(monitor.running ? "Pausar" : "Empezar") {
                        if monitor.running { monitor.stop() } else { monitor.start() }
                    }
                    .disabled(!monitor.ready)
                    Text("Cámara: \(monitor.cameraName)").foregroundStyle(.secondary)
                    Spacer()
                    Text(monitor.status).foregroundStyle(.secondary).lineLimit(2)
                }
                if let latest = monitor.latest {
                    HStack {
                        Circle().fill(actColors[latest.act] ?? .gray).frame(width: 14, height: 14)
                        Text(actNames[latest.act] ?? latest.act).font(.title2)
                        if let confidence = latest.confidence {
                            Text(String(format: "%.0f %%", 100 * confidence)).foregroundStyle(.secondary)
                        }
                    }
                }
                ForEach(monitor.alerts, id: \.identifier) { alert in
                    Label("\(alert.title): \(alert.body)", systemImage: "exclamationmark.triangle").foregroundStyle(.orange)
                }
            }
            DayPanel(records: monitor.today).frame(width: 320)
        }
        .padding(16)
    }
}

/// Resumen del día y línea de tiempo de 24 h.
struct DayPanel: View {
    let records: [WindowRecord]

    var body: some View {
        let summary = ActivitySummary(records)
        VStack(alignment: .leading, spacing: 10) {
            Text("Hoy").font(.headline)
            Timeline(records: records).frame(height: 44)
            HStack {
                Text("0 h")
                Spacer()
                Text("12 h")
                Spacer()
                Text("24 h")
            }.font(.caption).foregroundStyle(.secondary)
            ForEach(["lying", "locomotion", "other", "eating", "not_visible", "no_camera"], id: \.self) { act in
                HStack {
                    Circle().fill(actColors[act] ?? .gray).frame(width: 10, height: 10)
                    Text(actNames[act] ?? act)
                    Spacer()
                    Text(minutes(summary.secondsByAct[act] ?? 0)).monospacedDigit()
                }
            }
            Divider()
            if let rate = summary.displacementPerMinute {
                Text(String(format: "Movimiento: %.2f tamaños corporales por minuto (provisional)", rate))
            }
            Text("Comer no es fiable todavía; el movimiento con cámara fija está sin validar. No es un diagnóstico.")
                .font(.caption).foregroundStyle(.secondary)
            Spacer()
        }
    }

    func minutes(_ seconds: Double) -> String {
        seconds >= 3600 ? String(format: "%d h %02d min", Int(seconds) / 3600, (Int(seconds) % 3600) / 60)
            : String(format: "%d min", Int(seconds.rounded()) / 60)
    }
}

struct Timeline: View {
    let records: [WindowRecord]

    var body: some View {
        Canvas { context, size in
            context.fill(Path(CGRect(origin: .zero, size: size)), with: .color(.gray.opacity(0.08)))
            for r in records {
                let start = Calendar.current.startOfDay(for: r.start)
                let x0 = r.start.timeIntervalSince(start) / 86400 * size.width
                let width = max(1, r.seconds / 86400 * size.width)
                context.fill(Path(CGRect(x: x0, y: 0, width: width, height: size.height)),
                             with: .color(actColors[r.act] ?? .gray))
            }
        }
    }
}

/// Imagen en directo de la cámara.
struct CameraPreview: NSViewRepresentable {
    let session: AVCaptureSession

    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        let layer = AVCaptureVideoPreviewLayer(session: session)
        layer.videoGravity = .resizeAspect
        view.layer = layer
        view.wantsLayer = true
        return view
    }

    func updateNSView(_ nsView: NSView, context: Context) {}
}
