import CoreML
import Foundation
import Testing

@testable import KokoroCoreML

/// [[KokoroTailKernel]] — CoreML을 우회하는 Tail(conv_post + iSTFT) 커널.
///
/// 첫 테스트는 모델 없이 돈다: weight.bin 포맷을 흉내 낸 임시 `.mlmodelc`를 만들어 파싱과 수식을
/// Double 정밀도의 순진한 구현과 대조한다. 나머지는 실제 모델이 있을 때(`KOKORO_7STAGE_MODELS`)
/// CoreML `KokoroTail.mlmodelc`(cpuOnly) 출력과 비교한다 — 기기에서 죽던 바로 그 경로의 정답이다.
@Suite("KokoroTailKernel")
struct TailKernelTests {
    private static var modelDir: URL? {
        ProcessInfo.processInfo.environment["KOKORO_7STAGE_MODELS"].map(URL.init(fileURLWithPath:))
    }

    // MARK: - 합성 모델(모델 파일 불필요)

    @Test("가짜 mlmodelc 파싱 + 순진한 구현과 일치")
    func matchesNaiveReference() throws {
        var rng = SplitMix64(seed: 0xC0FFEE)
        let conv = (0..<(22 * 128 * 7)).map { _ in rng.nextFloat(in: -0.2...0.2) }
        let bias = (0..<22).map { _ in rng.nextFloat(in: -0.5...0.5) }
        let real = (0..<(11 * 20)).map { _ in rng.nextFloat(in: -0.1...0.1) }
        let imag = (0..<(11 * 20)).map { _ in rng.nextFloat(in: -0.1...0.1) }
        let dir = try Self.writeFakeModel(conv: conv, bias: bias, real: real, imag: imag)
        defer { try? FileManager.default.removeItem(at: dir) }

        let kernel = try KokoroTailKernel(modelDirectory: dir)

        for L in [1, 7, 100, 241, 1201] {
            let x = try MLMultiArray(shape: [1, 128, NSNumber(value: L)], dataType: .float32)
            let xp = x.dataPointer.assumingMemoryBound(to: Float.self)
            for i in 0..<(128 * L) { xp[i] = rng.nextFloat(in: -2...2) }
            let input = (0..<(128 * L)).map { xp[$0] }

            let got = try kernel.run(xPre: x)
            let want = Self.naiveTail(input, L: L, conv: conv, bias: bias, real: real, imag: imag)
            #expect(got.count == 5 * (L - 1), "L=\(L)")
            let scale = max(1e-6, want.map { abs($0) }.max() ?? 0)
            let maxErr = zip(got, want).map { abs(Double($0) - $1) }.max() ?? 0
            #expect(maxErr <= scale * 5e-5, "L=\(L) maxErr=\(maxErr) scale=\(scale)")
        }
    }

    @Test("fp16 입력도 같은 결과")
    func acceptsFloat16Input() throws {
        var rng = SplitMix64(seed: 42)
        let conv = (0..<(22 * 128 * 7)).map { _ in rng.nextFloat(in: -0.2...0.2) }
        let bias = (0..<22).map { _ in rng.nextFloat(in: -0.5...0.5) }
        let real = (0..<(11 * 20)).map { _ in rng.nextFloat(in: -0.1...0.1) }
        let imag = (0..<(11 * 20)).map { _ in rng.nextFloat(in: -0.1...0.1) }
        let dir = try Self.writeFakeModel(conv: conv, bias: bias, real: real, imag: imag)
        defer { try? FileManager.default.removeItem(at: dir) }
        let kernel = try KokoroTailKernel(modelDirectory: dir)

        let L = 361
        let half = try MLMultiArray(shape: [1, 128, NSNumber(value: L)], dataType: .float16)
        let full = try MLMultiArray(shape: [1, 128, NSNumber(value: L)], dataType: .float32)
        let hp = half.dataPointer.assumingMemoryBound(to: Float16.self)
        let fp = full.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<(128 * L) {
            let v = Float16(rng.nextFloat(in: -2...2))
            hp[i] = v
            fp[i] = Float(v)
        }
        let a = try kernel.run(xPre: half)
        let b = try kernel.run(xPre: full)
        #expect(a.count == b.count)
        #expect(zip(a, b).allSatisfy { abs($0 - $1) <= 1e-5 })
    }

    @Test("모델 파일이 없으면 modelsNotAvailable")
    func missingModelThrows() {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        #expect(throws: KokoroError.self) { try KokoroTailKernel(modelDirectory: dir) }
    }

    // MARK: - 실제 모델(CoreML Tail과 대조)

    @Test("CoreML KokoroTail(cpuOnly)과 일치", .enabled(if: modelDir != nil))
    func matchesCoreMLTail() throws {
        let dir = Self.modelDir!
        let kernel = try KokoroTailKernel(modelDirectory: dir)
        let config = MLModelConfiguration()
        config.computeUnits = .cpuOnly
        let model = try MLModel(contentsOf: dir.appendingPathComponent("KokoroTail.mlmodelc"), configuration: config)

        var rng = SplitMix64(seed: 7)
        // 크래시 로그의 실제 청크 길이(120·D+1)와 짧은 길이를 섞는다.
        for L in [3601, 14401, 34561, 37441, 72961] {
            let x = try MLMultiArray(shape: [1, 128, NSNumber(value: L)], dataType: .float32)
            let xp = x.dataPointer.assumingMemoryBound(to: Float.self)
            for i in 0..<(128 * L) { xp[i] = rng.nextFloat(in: -2...2) }

            let t0 = CFAbsoluteTimeGetCurrent()
            let got = try kernel.run(xPre: x)
            let kernelMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000

            let t1 = CFAbsoluteTimeGetCurrent()
            let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["x_pre": x]))
            let coreMLMs = (CFAbsoluteTimeGetCurrent() - t1) * 1000
            let want = MLArrayHelpers.extractFloats(from: try MLArrays.require(out, "audio"))

            #expect(got.count == want.count, "L=\(L)")
            let scale = want.map { abs($0) }.max() ?? 1
            let maxErr = zip(got, want).map { abs($0 - $1) }.max() ?? 0
            #expect(maxErr <= scale * 2e-5, "L=\(L) maxErr=\(maxErr) scale=\(scale)")
            print(String(format: "L=%d  kernel %.1fms  coreml %.1fms  maxErr %.2e (scale %.2f)", L, kernelMs, coreMLMs, maxErr, scale))
        }
    }

    // MARK: - Helpers

    /// model.mil의 관련 부분과 coremltools blob 포맷의 weight.bin을 가진 가짜 `KokoroTail.mlmodelc`.
    private static func writeFakeModel(conv: [Float], bias: [Float], real: [Float], imag: [Float]) throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("tailkernel-\(UUID().uuidString)")
        let model = dir.appendingPathComponent("KokoroTail.mlmodelc")
        try FileManager.default.createDirectory(at: model.appendingPathComponent("weights"), withIntermediateDirectories: true)

        var blob = Data(count: 64)  // 파일 헤더
        func append(_ values: [Float]) -> Int {
            let metaOffset = blob.count
            var meta = Data(count: 64)
            meta.replaceSubrange(0..<4, with: withUnsafeBytes(of: UInt32(0xDEAD_BEEF).littleEndian) { Data($0) })
            meta.replaceSubrange(4..<8, with: withUnsafeBytes(of: UInt32(2).littleEndian) { Data($0) })
            meta.replaceSubrange(8..<16, with: withUnsafeBytes(of: UInt64(values.count * 4).littleEndian) { Data($0) })
            meta.replaceSubrange(16..<24, with: withUnsafeBytes(of: UInt64(metaOffset + 64).littleEndian) { Data($0) })
            blob.append(meta)
            values.withUnsafeBufferPointer { blob.append(Data(buffer: $0)) }
            let padding = (64 - blob.count % 64) % 64
            blob.append(Data(count: padding))
            return metaOffset
        }
        let biasOff = append(bias)
        let realOff = append(real)
        let imagOff = append(imag)
        let convOff = append(conv)
        try blob.write(to: model.appendingPathComponent("weights/weight.bin"))

        func const(_ name: String, _ shape: String, _ offset: Int) -> String {
            "            tensor<fp32, [\(shape)]> \(name) = const()[name = tensor<string, []>(\"\(name)\"), val = tensor<fp32, [\(shape)]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/weight.bin\"), offset = tensor<uint64, []>(\(offset))))];"
        }
        let mil = """
            program(1.0)
            {
                func main<ios17>(tensor<fp32, [1, 128, ?]> x_pre) {
            \(const("conv_post_bias", "22", biasOff))
            \(const("stft_deconv_real_weight", "11, 1, 20", realOff))
            \(const("stft_deconv_imag_weight", "11, 1, 20", imagOff))
            \(const("weight_1", "22, 128, 7", convOff))
                        tensor<fp32, [1, 22, ?]> x = conv(bias = conv_post_bias, dilations = x_dilations_0, groups = x_groups_0, pad = x_pad_0, pad_type = x_pad_type_0, strides = x_strides_0, weight = weight_1, x = x_pre)[name = tensor<string, []>("x")];
                        tensor<fp32, [1, 1, ?]> var_71 = conv_transpose(dilations = var_71_dilations_0, groups = var_71_groups_0, pad = var_71_pad_0, pad_type = var_71_pad_type_0, strides = var_71_strides_0, weight = stft_deconv_real_weight, x = input_1)[name = tensor<string, []>("op_71")];
                        tensor<fp32, [1, 1, ?]> var_83 = conv_transpose(dilations = var_83_dilations_0, groups = var_83_groups_0, pad = var_83_pad_0, pad_type = var_83_pad_type_0, strides = var_83_strides_0, weight = stft_deconv_imag_weight, x = input)[name = tensor<string, []>("op_83")];
                        tensor<fp32, [1, 1, ?]> waveform = sub(x = var_71, y = var_83)[name = tensor<string, []>("waveform")];
                } -> (audio);
            }
            """
        try mil.write(to: model.appendingPathComponent("model.mil"), atomically: true, encoding: .utf8)
        return dir
    }

    /// Double 정밀도의 순진한 Tail 구현(루프 그대로).
    private static func naiveTail(_ x: [Float], L: Int, conv: [Float], bias: [Float], real: [Float], imag: [Float]) -> [Double] {
        var y = [Double](repeating: 0, count: 22 * L)
        for c in 0..<22 {
            for i in 0..<L {
                var acc = Double(bias[c])
                for ic in 0..<128 {
                    for k in 0..<7 {
                        let j = i + k - 3
                        guard j >= 0, j < L else { continue }
                        acc += Double(conv[(c * 128 + ic) * 7 + k]) * Double(x[ic * L + j])
                    }
                }
                y[c * L + i] = acc
            }
        }
        var wave = [Double](repeating: 0, count: (L - 1) * 5 + 20)
        for f in 0..<11 {
            for i in 0..<L {
                let mag = exp(y[f * L + i])
                let ph = sin(y[(11 + f) * L + i])
                let re = mag * cos(ph)
                let im = mag * sin(ph)
                for t in 0..<20 {
                    wave[i * 5 + t] += re * Double(real[f * 20 + t]) - im * Double(imag[f * 20 + t])
                }
            }
        }
        return Array(wave[10..<(wave.count - 10)])
    }
}

/// 테스트 재현용 결정적 난수.
private struct SplitMix64 {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    mutating func nextFloat(in range: ClosedRange<Float>) -> Float {
        let unit = Float(next() >> 40) / Float(1 << 24)
        return range.lowerBound + unit * (range.upperBound - range.lowerBound)
    }
}
