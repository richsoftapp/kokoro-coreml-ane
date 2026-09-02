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
    /// 한 체인(청크)이 낼 수 있는 최대 프레임 수. Prosody(`en` [1,640,L])·Vocoder(`asr` [1,512,L])의
    /// 변환 시 선언한 shape range 상한(L ≤ 2000 = 50초)이다. 이를 넘기면 CoreML이 입력 shape 오류를
    /// 던져 청크 전체가 실패하므로, `pred_dur` 합이 넘으면 [[clampDurations]]가 비례 축소한다.
    static let maxFrames = 2000
    /// 토큰 하나의 duration 상한(프레임). 모델의 duration head는 sigmoid 50개 합이라 speed 1.0에서
    /// 50, 최저 speed 0.1에서도 500을 넘을 수 없다 — 그 이상은 fp16 오버플로·NaN 같은 수치 이상이다.
    static let maxFramesPerToken = 800

    private let albert: MLModel
    private let postAlbert: MLModel
    private let alignment: MLModel
    private let prosody: MLModel
    private let noise: MLModel
    private let vocoder: MLModel
    private let tail: TailStage

    /// Tail(conv_post + iSTFT) 실행 경로.
    ///
    /// 기본은 [[KokoroTailKernel]](Accelerate) — iOS 26.6.x libBNNS의 SME2 conv 워크스페이스 오버플로우로
    /// A19 기기에서 매번 죽던 CoreML/BNNS 경로를 통째로 우회한다. 가중치 파싱이 실패하는 경우에만
    /// (모델 파일이 바뀐 경우) 예전 CoreML 모델로 물러난다.
    private enum TailStage {
        case kernel(KokoroTailKernel)
        case coreML(MLModel)
    }

    /// Tail이 CoreML 대신 Accelerate 커널로 도는지. 앱이 로드 로그에 남길 수 있게 노출한다.
    public var usesAccelerateTail: Bool {
        if case .kernel = tail { return true }
        return false
    }

    private let g2p: EnglishG2P
    private let tokenizer: Tokenizer
    private let voiceStore: VoiceStore

    public var availableVoices: [String] { voiceStore.availableVoices }
    /// 테스트용: 그룹 예산 보정을 위해 음소 길이를 재려고 G2P를 노출한다.
    var g2pForTesting: EnglishG2P { g2p }

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
        /// Tail은 기본적으로 CoreML을 거치지 않으므로([[KokoroTailKernel]]) 폴백 경로에만 적용된다.
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
        // Tail은 CoreML을 거치지 않는다(TailStage 참조). 커널이 못 뜨면 예전 경로로 물러난다.
        if let kernel = try? KokoroTailKernel(modelDirectory: modelDirectory) {
            self.tail = .kernel(kernel)
        } else {
            self.tail = .coreML(try load("KokoroTail", placement.tail))
        }

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
        // speed 0·음수·NaN은 PostAlbert 안의 나눗셈을 inf/NaN으로 만들어 duration이 깨진다. 여기서 막는다.
        let speed = Self.sanitizedSpeed(speed)
        // G2P의 두 번째 반환값(토큰)을 **버리지 않는다** — 기존 `Phonemizer` 프로토콜 경로가
        // `let (phonemes, _) = …`로 흘려보내던 그 값이 단어 타임스탬프의 절반이다.
        let (phonemes, tokens) = g2p.phonemize(text: text)

        var samples: [Float] = []
        var timestamps: [KokoroTokenTimestamp] = []
        var offset: Double = 0

        for group in Self.groupTokens(tokens, budget: Self.tokenBudget(forSpeed: speed)) {
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

    /// 합성 속도를 유효 범위로 맞춘다. NaN·inf·0 이하는 1.0으로, 그 외는 [0.1, 10]으로 자른다.
    static func sanitizedSpeed(_ speed: Float) -> Float {
        guard speed.isFinite, speed > 0 else { return 1.0 }
        return min(max(speed, 0.1), 10.0)
    }

    /// 속도에 따른 그룹 토큰 예산. 느린 속도는 토큰당 프레임이 늘어(duration ∝ 1/speed) 510토큰
    /// 그룹이 [[maxFrames]]를 넘길 수 있다 — 실측 평균 2.2프레임/음소(speed 1.0)·4.4(speed 0.5)라
    /// 510토큰은 speed 0.5에서 ~2240프레임이다. 넘기면 CoreML은 오류 없이 **2000프레임에서 잘라**
    /// 그룹 뒷부분 단어가 소리 없이 빠졌다(macOS 26 실측: 0.5배속 965음소 → 정확히 4000프레임).
    /// 예산을 속도에 비례해 줄여(0.5 → 318토큰 ≈ 1400프레임) 그룹이 상한 안에 들어오게 한다
    /// (그래도 넘기면 [[clampDurations]]가 잡지만, 그건 그 그룹만 조금 빨라지는 최후 방어다).
    static func tokenBudget(forSpeed speed: Float) -> Int {
        let full = maxTokens - 2
        guard speed < 0.8 else { return full }
        return max(64, Int(Double(full) * Double(speed) / 0.8))
    }

    /// PostAlbert의 duration 출력을 `pred_dur`로 바꾼다. `max(1, round(·))`(MLX 백엔드의 "clamped to
    /// minimum of 1 frame")에 세 가지 방어를 얹는다:
    ///  · NaN·inf는 1로 — `Int(Float.nan)`은 Swift 런타임 트랩이다(합성마다 죽는 크래시 클래스).
    ///  · 토큰당 [[maxFramesPerToken]] 상한 — 수치 이상 값이 수십 분짜리 프레임 축을 만들지 않게.
    ///  · 합이 `maxFrames`를 넘으면 비례 축소(각 토큰 최소 1 유지) — 넘기면 다음 스테이지의 CoreML
    ///    shape range 검사가 실패해 청크 전체가 throw 된다. 축소는 그 그룹만 조금 빨라질 뿐 소리는 난다.
    static func clampDurations(_ raw: [Float], maxFrames: Int = maxFrames) -> [Int] {
        var pred = raw.map { value -> Int in
            guard value.isFinite else { return 1 }
            return min(max(1, Int(value.rounded())), maxFramesPerToken)
        }
        var total = pred.reduce(0, +)
        guard total > maxFrames, pred.count <= maxFrames else { return pred }
        let scale = Double(maxFrames) / Double(total)
        pred = pred.map { max(1, Int((Double($0) * scale).rounded(.down))) }
        total = pred.reduce(0, +)
        // 바닥값 1로 올라간 항들 때문에 아직 넘으면 가장 긴 토큰부터 1프레임씩 깎는다(토큰 수 < maxFrames라 항상 끝난다).
        while total > maxFrames {
            var largest = 0
            for i in pred.indices where pred[i] > pred[largest] { largest = i }
            guard pred[largest] > 1 else { break }
            pred[largest] -= 1
            total -= 1
        }
        return pred
    }

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

        // pred_dur = max(1, round(duration)) + NaN·상한·프레임 합 방어([[clampDurations]]).
        let durations = MLArrayHelpers.extractFloats(from: durationArr, maxCount: T)
        // 출력이 토큰 수보다 짧으면 아래 `pred_dur` 배열 뒤가 초기화되지 않은 채 Alignment로 넘어간다.
        guard durations.count == T else {
            throw KokoroError.inferenceFailed("duration count \(durations.count) != token count \(T)")
        }
        let predDur = Self.clampDurations(durations)

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

        // 7. Tail — fp32 conv_post + iSTFT. 기본은 Accelerate 커널(x_pre fp16을 직접 받는다).
        let audio: [Float]
        switch tail {
        case .kernel(let kernel):
            audio = try kernel.run(xPre: xPre)
        case .coreML(let model):
            let o7 = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: [
                "x_pre": try MLArrays.cast(xPre, to: .float32),
            ]))
            audio = MLArrayHelpers.extractFloats(from: try MLArrays.require(o7, "audio"))
        }

        // 오디오 길이는 5(L−1) = duration 합 × 600이어야 한다. 혹시 더 길면 duration 합으로 자른다.
        let valid = min(predDur.reduce(0, +) * Self.hopSize, audio.count)
        return ChainOutput(
            samples: valid == audio.count ? audio : Array(audio.prefix(valid)),
            predDur: predDur
        )
    }
}

// MARK: - MLMultiArray 생성 헬퍼

enum MLArrays {
    /// 값 개수가 shape 크기와 정확히 같아야 한다. 많으면 MLMultiArray 버퍼 밖으로 쓰고(힙 손상),
    /// 적으면 뒤가 초기화되지 않은 채 모델로 들어간다. 둘 다 조용히 넘기지 않고 throw 한다.
    private static func checkCount(_ count: Int, shape: [Int]) throws {
        let expected = shape.reduce(1, *)
        guard count == expected, count > 0 else {
            throw KokoroError.inferenceFailed("MLMultiArray fill: \(count) values for shape \(shape)")
        }
    }

    static func int32(_ values: [Int32], shape: [Int]) throws -> MLMultiArray {
        try checkCount(values.count, shape: shape)
        let a = try MLMultiArray(shape: shape.map(NSNumber.init(value:)), dataType: .int32)
        let ptr = a.dataPointer.assumingMemoryBound(to: Int32.self)
        values.withUnsafeBufferPointer { ptr.update(from: $0.baseAddress!, count: values.count) }
        return a
    }

    static func float32(_ values: [Float], shape: [Int]) throws -> MLMultiArray {
        try checkCount(values.count, shape: shape)
        let a = try MLMultiArray(shape: shape.map(NSNumber.init(value:)), dataType: .float32)
        let ptr = a.dataPointer.assumingMemoryBound(to: Float.self)
        values.withUnsafeBufferPointer { ptr.update(from: $0.baseAddress!, count: values.count) }
        return a
    }

    static func float16(_ values: [Float], shape: [Int]) throws -> MLMultiArray {
        try checkCount(values.count, shape: shape)
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
