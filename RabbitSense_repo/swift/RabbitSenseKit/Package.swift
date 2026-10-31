// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "RabbitSenseKit",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "RabbitSenseKit", targets: ["RabbitSenseKit"]),
        .executable(name: "RabbitSenseApp", targets: ["RabbitSenseApp"]),
    ],
    targets: [
        .target(
            name: "RabbitSenseKit",
            resources: [.copy("Resources/ytactos_modelo.json")]
        ),
        .executableTarget(
            name: "RabbitSenseApp",
            dependencies: ["RabbitSenseKit"]
        ),
        .testTarget(
            name: "RabbitSenseKitTests",
            dependencies: ["RabbitSenseKit"],
            exclude: ["Fixtures"]
        ),
    ]
)
