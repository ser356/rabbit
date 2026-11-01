import AVFoundation
import CoreImage
import RabbitSenseKit

/// Cámara del Mac: prefiere el iPhone (Continuity Camera), luego una externa y por último la integrada. Entrega
/// fotogramas reducidos a lado largo 640, a ~15 por segundo, sin guardar nada en disco.
final class Camera: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {
    let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "rabbitsense.camera")
    private let context = CIContext()
    private var lastTime = -Double.infinity
    var onFrame: ((RGBImage, CGImage, Date) -> Void)?
    private(set) var deviceName = "sin cámara"

    static func devices() -> [AVCaptureDevice] {
        AVCaptureDevice.DiscoverySession(deviceTypes: [.continuityCamera, .external, .builtInWideAngleCamera],
                                         mediaType: .video, position: .unspecified).devices
    }

    /// El iPhone si está disponible; si no, la primera cámara.
    static func preferred() -> AVCaptureDevice? {
        let all = devices()
        return all.first { $0.deviceType == .continuityCamera } ?? all.first
    }

    func start(device: AVCaptureDevice) throws {
        session.beginConfiguration()
        session.inputs.forEach(session.removeInput)
        session.outputs.forEach(session.removeOutput)
        session.sessionPreset = .high
        let input = try AVCaptureDeviceInput(device: device)
        guard session.canAddInput(input) else { throw RabbitSenseError.model("No se puede usar \(device.localizedName)") }
        session.addInput(input)
        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: queue)
        guard session.canAddOutput(output) else { throw RabbitSenseError.model("No se puede leer la cámara") }
        session.addOutput(output)
        session.commitConfiguration()
        deviceName = device.localizedName
        queue.async { self.session.startRunning() }
    }

    func stop() {
        queue.async { self.session.stopRunning() }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        let time = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sampleBuffer))
        guard time - lastTime >= 1.0 / 15 - 0.005, let buffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        lastTime = time
        let image = CIImage(cvPixelBuffer: buffer)
        let scale = min(1, 640 / max(image.extent.width, image.extent.height))
        let small = image.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        guard let cgImage = context.createCGImage(small, from: small.extent),
              let rgb = try? rgbImage(cgImage) else { return }
        onFrame?(rgb, cgImage, Date())
    }
}
