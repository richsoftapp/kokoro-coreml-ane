import CoreML
import Foundation

/// Kokoro를 **7개 CoreML 스테이지**로 나눠 실행하는 엔진.
///
/// 기존 [[KokoroEngine]]은 frontend/backend 2개 모놀리식 모델을 쓴다. 그래프 하나에 컴퓨트 유닛이
/// 하나씩 걸리므로, 그 안의 op 하나가 ANE 컴파일에 실패하면 세그먼트 전체가 CPU float32로 떨어져
/// 추론 1회에 ~1GB를 쓴다. 스테이지를 쪼개면 **ANE가 못 먹는 부분만 격리**해 나머지를 살릴 수 있다.
///
/// 모델은 `laishere/kokoro-coreml`(Apache-2.0)의 변환 파이프라인 산출물이고, G2P·토크나이저·
/// 보이스 스토어는 이 패키지의 것을 그대로 쓴다 — vocab이 양쪽 114개 완전 일치하고,
/// `Tokenizer.encode`가 붙이는 BOS/EOS도 변환 스크립트의 `[0, …, 0]`과 같다.
///
/// 기존 엔진과 달리 **단어 단위 타임스탬프**를 낸다([[TokenTimestampPredictor]]). duration predictor의
/// 출력(`pred_dur`)이 Swift 레벨로 나오고, `EnglishG2P`가 만든 `[MToken]`이 그대로 살아 있어서다.
public final class KokoroSevenStageEngine: @unchecked Sendable {
    public static let sampleRate = 24_000
    /// ALBERT의 포지션 임베딩 상한(BOS/EOS 포함).
    static let maxTokens = 512
    /// duration 1프레임 = 오디오 600샘플(24 kHz ÷ 40 fps).
    static let hopSize = 600

    private let albert: MLModel
    private let postAlbert: MLModel
    private let alignment: MLModel
    private let prosody: MLModel
    private let noise: MLModel
    private let vocoder: MLModel
    private let tail: MLModel

    private let g2p: EnglishG2P
    private let tokenizer: Tokenizer
    private let voiceStore: VoiceStore

    public var availableVoices: [String] { voiceStore.availableVoices }

    /// 스테이지별 컴퓨트 유닛.
    ///
    /// `postAlbert`만 `.cpuOnly`인 것이 중요하다 — ANE(`.cpuAndNeuralEngine`)로 돌리면 duration을
    /// **틀리게** 계산한다(macOS 26.5.1에서 48개 토큰 중 37개만 일치, 합계 120→134). 양자화 문제가
    /// 아니다: 같은 int8pal 가중치로 CPU는 PyTorch와 48/48 정확히 일치한다. 게다가 이 스테이지는
    /// 작은 LSTM이라 CPU가 **가장 빠르기까지 하다**(1.9ms vs GPU 9.4ms). 잃는 게 없다.
    /// duration이 틀리면 오디오뿐 아니라 단어 타임스탬프도 함께 틀어지므로 우회 불가.
    ///
    /// 기본값은 **Metal을 전혀 쓰지 않는다**(`.all` 없음). 백그라운드에서 GPU 제출이 거부되는 걸
    /// 피하는 게 이 엔진을 도입한 이유이기 때문이다. 대가는 작다 — prosody/noise/tail을 `.all`에서
    /// `.cpuAndNeuralEngine`으로 내려도 mel_corr 0.9936 → 0.9934, 체인 63.7ms → 68.0ms(약 6%)다.
    public struct StagePlacement: Sendable {
        public var albert: MLComputeUnits = .cpuAndNeuralEngine
        public var postAlbert: MLComputeUnits = .cpuOnly
        public var alignment: MLComputeUnits = .cpuAndNeuralEngine
        public var prosody: MLComputeUnits = .cpuAndNeuralEngine
        public var noise: MLComputeUnits = .cpuAndNeuralEngine
        public var vocoder: MLComputeUnits = .cpuAndNeuralEngine
        public var tail: MLComputeUnits = .cpuAndNeuralEngine

        public init() {}

        /// GPU를 허용해 약간 더 빠른 배치(전경 전용). 백그라운드 재생에는 쓰지 않는다.
        public static var allowingGPU: StagePlacement {
            var p = StagePlacement()
            p.prosody = .all
            p.noise = .all
            p.tail = .all
            return p
        }
    }

    /// 컴파일된 `.mlmodelc` 7개가 들어 있는 디렉터리에서 로드한다.
    public init(
        modelDirectory: URL,
        voicesDirectory: URL,
        british: Bool = false,
        placement: StagePlacement = StagePlacement()
    ) throws {
        func load(_ name: String, _ units: MLComputeUnits) throws -> MLModel {
            let url = modelDirectory.appendingPathComponent("\(name).mlmodelc")
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw KokoroError.modelsNotAvailable(url)
            }
            let config = MLModelConfiguration()
            config.computeUnits = units
            return try MLModel(contentsOf: url, configuration: config)
        }
        // 스테이지마다 한 번씩만 로드한다. 같은 모델을 한 프로세스에서 여러 컴퓨트 유닛으로 다시
        // 로드하면 E5RT가 미초기화 reps로 죽는다(`Tile: Shape deduction failed as reps[0]=<garbage>`).
        self.albert = try load("KokoroAlbert", placement.albert)
        self.postAlbert = try load("KokoroPostAlbert", placement.postAlbert)
        self.alignment = try load("KokoroAlignment", placement.alignment)
        self.prosody = try load("KokoroProsody", placement.prosody)
        self.noise = try load("KokoroNoise", placement.noise)
        self.vocoder = try load("KokoroVocoder", placement.vocoder)
        self.tail = try load("KokoroTail", placement.tail)

        self.g2p = EnglishG2P(british: british)
        self.tokenizer = try Tokenizer.loadFromBundle()
        self.voiceStore = try VoiceStore(directory: voicesDirectory)
    }

    // MARK: - Synthesis

    public struct Result: Sendable {
        public let samples: [Float]
        public let phonemes: String
        /// 단어 단위 시작/끝 시각(초). 이 결과 오디오 전체 기준.
        public let timestamps: [KokoroTokenTimestamp]
        public let synthesisTime: TimeInterval
        public var duration: TimeInterval { Double(samples.count) / Double(sampleRate) }
        public var realTimeFactor: Double { synthesisTime > 0 ? duration / synthesisTime : 0 }
    }

    public func synthesize(text: String, voice: String, speed: Float = 1.0) throws -> Result {
        let start = CFAbsoluteTimeGetCurrent()
        // G2P의 두 번째 반환값(토큰)을 **버리지 않는다** — 기존 `Phonemizer` 프로토콜 경로가
        // `let (phonemes, _) = …`로 흘려보내던 그 값이 단어 타임스탬프의 절반이다.
        let (phonemes, tokens) = g2p.phonemize(text: text)

        var samples: [Float] = []
        var timestamps: [KokoroTokenTimestamp] = []
        var offset: Double = 0

        for group in Self.groupTokens(tokens, budget: Self.maxTokens - 2) {
            try autoreleasepool {
                let ids = tokenizer.encode(group.phonemes, maxLength: Self.maxTokens)
                guard ids.count > 2 else { return }
                let refS = try voiceStore.embedding(for: voice, tokenCount: ids.count - 2)
                let chunk = try runChain(tokenIds: ids, refS: refS, speed: speed)

                timestamps.append(
                    contentsOf: TokenTimestampPredictor.timestamps(
                        from: group.tokens, predDur: chunk.predDur, offset: offset
                    )
                )
                samples.append(contentsOf: chunk.samples)
                offset += Double(chunk.samples.count) / Double(Self.sampleRate)
            }
        }

        return Result(
            samples: samples,
            phonemes: phonemes,
            timestamps: timestamps,
            synthesisTime: CFAbsoluteTimeGetCurrent() - start
        )
    }

    // MARK: - Chunking

    struct TokenGroup {
        let tokens: [MToken]
        let phonemes: String
    }

    /// `[MToken]`을 **단어 경계에서** 토큰 예산 이하로 묶는다.
    ///
    /// 음소 문자열을 먼저 이어붙인 뒤 자르면 단어 중간에서 끊겨 타임스탬프가 어긋난다. 토큰 단위로
    /// 묶어야 각 그룹의 `[MToken]`과 그 그룹 오디오의 `pred_dur`가 1:1로 맞는다.
    static func groupTokens(_ tokens: [MToken], budget: Int) -> [TokenGroup] {
        var groups: [TokenGroup] = []
        var current: [MToken] = []
        var text = ""

        func flush() {
            guard !current.isEmpty, !text.isEmpty else { current = []; text = ""; return }
            groups.append(TokenGroup(tokens: current, phonemes: text))
            current = []
            text = ""
        }

        for token in tokens {
            let piece = (token.phonemes ?? "") + token.whitespace
            if !text.isEmpty, text.count + piece.count > budget { flush() }
            current.append(token)
            text += piece
        }
        flush()
        return groups
    }

    // MARK: - 7-stage chain

    private struct ChainOutput {
        let samples: [Float]
        let predDur: [Int]
    }

    private func runChain(tokenIds: [Int], refS: [Float], speed: Float) throws -> ChainOutput {
        let T = tokenIds.count
        let ids = try MLArrays.int32(tokenIds.map(Int32.init), shape: [1, T])
        let mask = try MLArrays.int32([Int32](repeating: 1, count: T), shape: [1, T])
        // ref_s는 256차원. 앞 128 = timbre(음색), 뒤 128 = style(운율). 변환 스크립트와 동일한 분할.
        let timbre = Array(refS.prefix(128))
        let style = Array(refS.dropFirst(128).prefix(128))
        let styleF16 = try MLArrays.float16(style, shape: [1, 128])
        let timbreF16 = try MLArrays.float16(timbre, shape: [1, 128])
        let timbreF32 = try MLArrays.float32(timbre, shape: [1, 128])

        // 1. ALBERT — 텍스트 인코더
        let o1 = try albert.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "input_ids": ids, "attention_mask": mask,
        ]))
        let bertDur = try MLArrays.require(o1, "bert_dur")

        // 2. PostAlbert — duration + d + t_en
        let o2 = try postAlbert.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "bert_dur": bertDur,
            "input_ids": ids,
            "style_s": styleF16,
            "speed": try MLArrays.float16([speed], shape: [1]),
            "attention_mask": mask,
        ]))
        let durationArr = try MLArrays.require(o2, "duration")
        let dArr = try MLArrays.require(o2, "d")
        let tEnArr = try MLArrays.require(o2, "t_en")

        // pred_dur = max(1, round(duration)). MLX 백엔드의 "clamped to minimum of 1 frame"과 동일.
        let durations = MLArrayHelpers.extractFloats(from: durationArr, maxCount: T)
        let predDur = durations.map { max(1, Int($0.rounded())) }

        // 3. Alignment — duration으로 프레임 축 전개
        let o3 = try alignment.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "pred_dur": try MLArrays.int32(predDur.map(Int32.init), shape: [1, T]),
            "d": dArr,
            "t_en": tEnArr,
        ]))
        let enArr = try MLArrays.require(o3, "en")
        let asrArr = try MLArrays.require(o3, "asr")

        // 4. Prosody — F0/N 곡선
        let o4 = try prosody.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "en": enArr, "style_s": styleF16,
        ]))
        let f0Arr = try MLArrays.require(o4, "F0")
        let nArr = try MLArrays.require(o4, "N")

        // 5. Noise — 하모닉 소스. 7스테이지 중 유일하게 fp32 입출력이라 경계에서 변환한다.
        let o5 = try noise.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "F0_curve": try MLArrays.cast(f0Arr, to: .float32),
            "style_timbre": timbreF32,
        ]))
        let xs0 = try MLArrays.require(o5, "x_source_0")
        let xs1 = try MLArrays.require(o5, "x_source_1")

        // 6. Vocoder — ANE 그래프를 유지하려 audio anchor도 내지만 버리고 x_pre만 쓴다.
        //
        // x_source_*는 Noise가 **Float32**로 내는데 Vocoder 입력은 **Float16**이다. CoreML은 이
        // 불일치를 자동 변환해 주지 않고 `Cannot retrieve vector from IRValue format …`으로 죽는다.
        // (Python 경로는 numpy `.astype(np.float16)`이 가려 줬다.) 스테이지 경계마다 선언된 타입으로
        // 맞춰 넘긴다.
        let o6 = try vocoder.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "asr": asrArr,
            "F0_curve": f0Arr,
            "N_pred": nArr,
            "x_source_0": try MLArrays.cast(xs0, to: .float16),
            "x_source_1": try MLArrays.cast(xs1, to: .float16),
            "style_timbre": timbreF16,
        ]))
        let xPre = try MLArrays.require(o6, "x_pre")

        // 7. Tail — fp32 conv_post + iSTFT
        let o7 = try tail.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "x_pre": try MLArrays.cast(xPre, to: .float32),
        ]))
        let audio = try MLArrays.require(o7, "audio")

        // 모델은 버킷 길이까지 패딩된 오디오를 낸다. duration 합이 실제 길이다.
        let valid = min(predDur.reduce(0, +) * Self.hopSize, audio.count)
        return ChainOutput(
            samples: MLArrayHelpers.extractFloats(from: audio, maxCount: valid),
            predDur: predDur
        )
    }
}

// MARK: - MLMultiArray 생성 헬퍼

enum MLArrays {
    static func int32(_ values: [Int32], shape: [Int]) throws -> MLMultiArray {
        let a = try MLMultiArray(shape: shape.map(NSNumber.init(value:)), dataType: .int32)
        let ptr = a.dataPointer.assumingMemoryBound(to: Int32.self)
        values.withUnsafeBufferPointer { ptr.update(from: $0.baseAddress!, count: values.count) }
        return a
    }

    static func float32(_ values: [Float], shape: [Int]) throws -> MLMultiArray {
        let a = try MLMultiArray(shape: shape.map(NSNumber.init(value:)), dataType: .float32)
        let ptr = a.dataPointer.assumingMemoryBound(to: Float.self)
        values.withUnsafeBufferPointer { ptr.update(from: $0.baseAddress!, count: values.count) }
        return a
    }

    static func float16(_ values: [Float], shape: [Int]) throws -> MLMultiArray {
        let a = try MLMultiArray(shape: shape.map(NSNumber.init(value:)), dataType: .float16)
        let ptr = a.dataPointer.assumingMemoryBound(to: Float16.self)
        for (i, v) in values.enumerated() { ptr[i] = Float16(v) }
        return a
    }

    /// 스테이지 출력 배열을 다음 스테이지가 선언한 dtype으로 맞춘다(이미 같으면 그대로 통과).
    ///
    /// CoreML은 입력 dtype 불일치를 자동 변환하지 않고 `Cannot retrieve vector from IRValue
    /// format …`류의 오류로 실패한다. 모델이 fp16/fp32를 섞어 쓰므로(Noise만 fp32) 경계에서 명시 변환이 필요하다.
    static func cast(_ array: MLMultiArray, to type: MLMultiArrayDataType) throws -> MLMultiArray {
        guard array.dataType != type else { return array }
        let shape = array.shape.map(\.intValue)
        let values = MLArrayHelpers.extractFloats(from: array)
        switch type {
        case .float16: return try float16(values, shape: shape)
        case .float32: return try float32(values, shape: shape)
        default: throw KokoroError.inferenceFailed("Unsupported cast target: \(type)")
        }
    }

    static func require(_ provider: MLFeatureProvider, _ name: String) throws -> MLMultiArray {
        guard let value = provider.featureValue(for: name)?.multiArrayValue else {
            throw KokoroError.inferenceFailed("Missing output: \(name)")
        }
        return value
    }
}
