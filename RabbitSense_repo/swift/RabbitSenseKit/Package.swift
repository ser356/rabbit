// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "RabbitSenseKit",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "RabbitSenseKit", targets: ["RabbitSenseKit"]),
    ],
    targets: [
        .target(
            name: "RabbitSenseKit",
            resources: [.copy("Resources/ytactos_modelo.json")]
        ),
        .testTarget(
            name: "RabbitSenseKitTests",
            dependencies: ["RabbitSenseKit"],
            exclude: ["Fixtures"]
        ),
    ]
)
