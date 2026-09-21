import Accelerate
import CoreML
import Foundation

/// KokoroTail 스테이지(conv_post → 진폭/위상 → iSTFT)를 **CoreML 없이 Accelerate로** 계산한다.
///
/// 이 스테이지만 CoreML을 우회하는 이유: iOS 26.6.x의 libBNNS SME2 conv 커널
/// (`bnns::graph::conv2d_params_S1xY<float,float>::repack_input`)이 128→22채널 conv_post를 돌릴 때
/// 자기 워크스페이스(352·L 바이트) **밖으로 쓴다**(크래시 로그 esr = Data Abort *write*, 폴트 주소 =
/// VM 영역 끝 + 1). 이 경로는 SME2가 있는 A19 계열(iPhone 17)에서만 선택되므로 그 기기에서는 청크
/// 길이와 무관하게 합성할 때마다 SIGSEGV였다(2.2(48) 크래시 10건 전부 동일 스택, L = 288~608 프레임).
/// ANE는 fp16 전용이라 이 fp32 스테이지는 항상 CPU(BNNS)로 떨어지고, 컴퓨트 유닛 설정으로는 피할 수 없다.
///
/// 가중치는 번들의 `KokoroTail.mlmodelc`(model.mil + weights/weight.bin)에서 그대로 읽는다 — 새 리소스도,
/// 모델 재변환도 없다. 계산은 model.mil 그래프와 1:1이다:
///
///     x     = conv1d(x_pre, W[22,128,7], b, pad 3, stride 1)          // [22, L]
///     mag   = exp(x[0..<11]);   ph = sin(x[11..<22])
///     real  = mag · cos(ph);    imag = mag · sin(ph)
///     wav   = convT(real, Wr[11,20], stride 5) − convT(imag, Wi[11,20], stride 5)   // [5(L−1)+20]
///     audio = wav[10 ..< −10]                                          // [5(L−1)]
///
/// conv는 탭(k)별 GEMM 7번(`W_k[22×128] · x_pad[128×L]`)으로, iSTFT는 GEMM 한 쌍(`[L×11]·[11×20]`)과
/// 홉 5 오버랩-애드로 푼다. 모두 libBLAS/vDSP/vForce라 libBNNS를 건드리지 않는다.
final class KokoroTailKernel: @unchecked Sendable {
    static let inChannels = 128
    static let outChannels = 22
    static let kernelSize = 7
    static let bins = 11
    static let frameSize = 20
    static let hop = 5
    /// iSTFT 출력 양끝에서 잘라내는 샘플 수(model.mil의 `slice_by_index [10, -10]`).
    static let edgeTrim = 10

    /// conv_post 가중치를 탭 우선으로 재배열한 것: `[k][outChannel][inChannel]` (7 × 22 × 128).
    private let convWeightByTap: [Float]
    /// conv_post 바이어스 `[22]`.
    private let convBias: [Float]
    /// iSTFT 실수부 기저 `[11][20]`(model.mil `stft_deconv_real_weight`).
    private let realBasis: [Float]
    /// iSTFT 허수부 기저 `[11][20]`(model.mil `stft_deconv_imag_weight`).
    private let imagBasis: [Float]

    /// `KokoroTail.mlmodelc`가 들어 있는 디렉터리에서 가중치를 읽는다.
    init(modelDirectory: URL) throws {
        let modelURL = modelDirectory.appendingPathComponent("KokoroTail.mlmodelc")
        let weights = try TailWeights(modelURL: modelURL)
        convBias = weights.bias
        realBasis = weights.real
        imagBasis = weights.imag

        // [22][128][7] → [7][22][128]: 탭별 GEMM이 연속 메모리를 읽게.
        let (co, ci, k) = (Self.outChannels, Self.inChannels, Self.kernelSize)
        var byTap = [Float](repeating: 0, count: k * co * ci)
        for c in 0..<co {
            for i in 0..<ci {
                for t in 0..<k {
                    byTap[(t * co + c) * ci + i] = weights.conv[(c * ci + i) * k + t]
                }
            }
        }
        convWeightByTap = byTap
    }

    /// `x_pre` `[1, 128, L]`(fp16 또는 fp32)에서 오디오 `[5(L−1)]`을 만든다.
    func run(xPre: MLMultiArray) throws -> [Float] {
        let shape = xPre.shape.map(\.intValue)
        guard shape.count == 3, shape[0] == 1, shape[1] == Self.inChannels, shape[2] >= 1 else {
            throw KokoroError.inferenceFailed("KokoroTail: unexpected x_pre shape \(shape)")
        }
        let L = shape[2]
        let strides = xPre.strides.map(\.intValue)
        let rowStride = strides[1]
        guard strides[2] == 1, rowStride >= L else {
            throw KokoroError.inferenceFailed("KokoroTail: unsupported x_pre strides \(strides)")
        }

        // 1. 양옆 3프레임 0 패딩한 fp32 입력 [128][L+6].
        let pad = Self.kernelSize / 2
        let paddedWidth = L + 2 * pad
        var padded = [Float](repeating: 0, count: Self.inChannels * paddedWidth)
        try padded.withUnsafeMutableBufferPointer { dst in
            var out = vImage_Buffer(
                data: dst.baseAddress! + pad, height: vImagePixelCount(Self.inChannels),
                width: vImagePixelCount(L), rowBytes: paddedWidth * MemoryLayout<Float>.stride
            )
            switch xPre.dataType {
            case .float16:
                var src = vImage_Buffer(
                    data: xPre.dataPointer, height: vImagePixelCount(Self.inChannels),
                    width: vImagePixelCount(L), rowBytes: rowStride * MemoryLayout<Float16>.stride
                )
                let err = vImageConvert_Planar16FtoPlanarF(&src, &out, vImage_Flags(kvImageNoFlags))
                guard err == kvImageNoError else {
                    throw KokoroError.inferenceFailed("KokoroTail: fp16→fp32 convert failed (\(err))")
                }
            case .float32:
                let src = xPre.dataPointer.assumingMemoryBound(to: Float.self)
                for c in 0..<Self.inChannels {
                    (dst.baseAddress! + c * paddedWidth + pad).update(from: src + c * rowStride, count: L)
                }
            default:
                throw KokoroError.inferenceFailed("KokoroTail: unsupported x_pre dtype \(xPre.dataType)")
            }
        }

        // 2. conv_post: y[22][L] = b + Σ_k W_k · x_pad[:, k ..< k+L].
        let co = Self.outChannels
        var y = [Float](repeating: 0, count: co * L)
        y.withUnsafeMutableBufferPointer { yp in
            for c in 0..<co {
                var b = convBias[c]
                vDSP_vfill(&b, yp.baseAddress! + c * L, 1, vDSP_Length(L))
            }
            padded.withUnsafeBufferPointer { xp in
                convWeightByTap.withUnsafeBufferPointer { wp in
                    for k in 0..<Self.kernelSize {
                        // 레거시 CBLAS 시그니처를 일부러 쓴다: ACCELERATE_NEW_LAPACK은 clang 매크로라
                        // SwiftPM 의존성에서 unsafeFlags 없이 켤 수 없다. 기능·성능은 동일하다.
                        cblas_sgemm(
                            CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            Int32(co), Int32(L), Int32(Self.inChannels),
                            1, wp.baseAddress! + k * co * Self.inChannels, Int32(Self.inChannels),
                            xp.baseAddress! + k, Int32(paddedWidth),
                            1, yp.baseAddress!, Int32(L)
                        )
                    }
                }
            }
        }

        // 3. 진폭·위상 → 실수/허수 스펙트럼 [11][L].
        let bins = Self.bins
        var n = Int32(bins * L)
        var mag = [Float](repeating: 0, count: bins * L)
        var phase = [Float](repeating: 0, count: bins * L)
        var real = [Float](repeating: 0, count: bins * L)
        var imag = [Float](repeating: 0, count: bins * L)
        y.withUnsafeBufferPointer { yp in
            vvexpf(&mag, yp.baseAddress!, &n)
            vvsinf(&phase, yp.baseAddress! + bins * L, &n)
        }
        vvcosf(&real, phase, &n)
        vvsinf(&imag, phase, &n)
        vDSP_vmul(mag, 1, real, 1, &real, 1, vDSP_Length(bins * L))
        vDSP_vmul(mag, 1, imag, 1, &imag, 1, vDSP_Length(bins * L))

        // 4. iSTFT: 프레임별 20샘플 창 G[L][20] = realᵀ·Wr − imagᵀ·Wi, 홉 5로 오버랩-애드.
        let frame = Self.frameSize
        var frames = [Float](repeating: 0, count: L * frame)
        frames.withUnsafeMutableBufferPointer { gp in
            cblas_sgemm(
                CblasRowMajor, CblasTrans, CblasNoTrans, Int32(L), Int32(frame), Int32(bins),
                1, real, Int32(L), realBasis, Int32(frame), 0, gp.baseAddress!, Int32(frame)
            )
            cblas_sgemm(
                CblasRowMajor, CblasTrans, CblasNoTrans, Int32(L), Int32(frame), Int32(bins),
                -1, imag, Int32(L), imagBasis, Int32(frame), 1, gp.baseAddress!, Int32(frame)
            )
        }
        let waveLength = (L - 1) * Self.hop + frame
        var wave = [Float](repeating: 0, count: waveLength)
        wave.withUnsafeMutableBufferPointer { wp in
            frames.withUnsafeBufferPointer { gp in
                for i in 0..<L {
                    let g = gp.baseAddress! + i * frame
                    let w = wp.baseAddress! + i * Self.hop
                    for t in 0..<frame { w[t] += g[t] }
                }
            }
        }

        // 5. 양끝 10샘플 트림 → 5(L−1) 샘플.
        let trim = Self.edgeTrim
        return Array(wave[trim..<(waveLength - trim)])
    }
}

// MARK: - 가중치 로딩

/// `KokoroTail.mlmodelc`의 model.mil에서 const 텐서의 BLOBFILE 오프셋을 찾고 weight.bin에서 읽는다.
///
/// weight.bin은 coremltools의 blob 저장 포맷이다: 각 텐서 앞에 64바이트 정렬 메타데이터
/// `{ sentinel 0xDEADBEEF: u32, dtype: u32, sizeInBytes: u64, dataOffset: u64 }`가 있고,
/// model.mil의 `offset`은 이 메타데이터의 위치다.
struct TailWeights {
    let conv: [Float]  // [22][128][7]
    let bias: [Float]  // [22]
    let real: [Float]  // [11][1][20]
    let imag: [Float]  // [11][1][20]

    private static let blobSentinel: UInt32 = 0xDEAD_BEEF
    private static let blobFloat32: UInt32 = 2

    init(modelURL: URL) throws {
        let milURL = modelURL.appendingPathComponent("model.mil")
        let binURL = modelURL.appendingPathComponent("weights/weight.bin")
        guard FileManager.default.fileExists(atPath: milURL.path),
            FileManager.default.fileExists(atPath: binURL.path)
        else { throw KokoroError.modelsNotAvailable(modelURL) }

        let mil = try String(contentsOf: milURL, encoding: .utf8)
        let blob = try Data(contentsOf: binURL)

        // conv 옵의 weight/bias 변수명은 변환기가 붙이므로(예: weight_1) 옵 정의에서 역추적한다.
        guard let convMatch = mil.firstMatch(of: /= conv\(bias = (\w+), [^\n]*?weight = (\w+), x = x_pre\)/) else {
            throw KokoroError.modelLoadFailed("KokoroTail: conv_post op not found in model.mil")
        }
        let biasName = String(convMatch.1)
        let weightName = String(convMatch.2)
        // conv_transpose 두 개는 `waveform = sub(x = real, y = imag)`의 피연산자 순서로 실수/허수를 가른다.
        guard let subMatch = mil.firstMatch(of: /= sub\(x = (\w+), y = (\w+)\)/) else {
            throw KokoroError.modelLoadFailed("KokoroTail: waveform sub op not found in model.mil")
        }
        func deconvWeight(producing output: Substring) throws -> String {
            let pattern = try Regex("\(output) = conv_transpose\\([^\\n]*?weight = (\\w+),")
            guard let m = mil.firstMatch(of: pattern), let name = m.output[1].substring else {
                throw KokoroError.modelLoadFailed("KokoroTail: conv_transpose producing \(output) not found")
            }
            return String(name)
        }
        let realName = try deconvWeight(producing: subMatch.1)
        let imagName = try deconvWeight(producing: subMatch.2)

        conv = try Self.tensor(named: weightName, expectedShape: [22, 128, 7], mil: mil, blob: blob)
        bias = try Self.tensor(named: biasName, expectedShape: [22], mil: mil, blob: blob)
        real = try Self.tensor(named: realName, expectedShape: [11, 1, 20], mil: mil, blob: blob)
        imag = try Self.tensor(named: imagName, expectedShape: [11, 1, 20], mil: mil, blob: blob)
    }

    private static func tensor(named name: String, expectedShape: [Int], mil: String, blob: Data) throws -> [Float] {
        let pattern = try Regex(
            "tensor<fp32, \\[([0-9, ]+)\\]> \(name) = const\\(\\)\\[[^\\n]*?BLOBFILE\\([^\\n]*?offset = tensor<uint64, \\[\\]>\\((\\d+)\\)"
        )
        guard let m = mil.firstMatch(of: pattern),
            let shapeText = m.output[1].substring, let offsetText = m.output[2].substring,
            let metaOffset = Int(offsetText)
        else { throw KokoroError.modelLoadFailed("KokoroTail: const \(name) not found in model.mil") }
        let shape = shapeText.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        guard shape == expectedShape else {
            throw KokoroError.modelLoadFailed("KokoroTail: \(name) shape \(shape) != \(expectedShape)")
        }

        guard metaOffset + 24 <= blob.count else {
            throw KokoroError.modelLoadFailed("KokoroTail: \(name) blob metadata out of range")
        }
        let sentinel = blob.readLE(UInt32.self, at: metaOffset)
        let dtype = blob.readLE(UInt32.self, at: metaOffset + 4)
        let size = Int(blob.readLE(UInt64.self, at: metaOffset + 8))
        let dataOffset = Int(blob.readLE(UInt64.self, at: metaOffset + 16))
        let count = shape.reduce(1, *)
        guard sentinel == blobSentinel, dtype == blobFloat32, size == count * 4, dataOffset + size <= blob.count else {
            throw KokoroError.modelLoadFailed(
                "KokoroTail: \(name) blob header invalid (sentinel \(String(sentinel, radix: 16)), dtype \(dtype), size \(size))"
            )
        }
        var values = [Float](repeating: 0, count: count)
        values.withUnsafeMutableBytes { dst in
            _ = blob.copyBytes(to: dst, from: dataOffset..<(dataOffset + size))
        }
        return values
    }
}

extension Data {
    fileprivate func readLE<T: FixedWidthInteger>(_: T.Type, at offset: Int) -> T {
        let raw = subdata(in: offset..<(offset + MemoryLayout<T>.size))
            .withUnsafeBytes { $0.loadUnaligned(as: T.self) }
        return T(littleEndian: raw)
    }
}
