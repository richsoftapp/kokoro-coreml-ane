import CoreML
import Foundation
import Testing

@testable import KokoroCoreML

/// 7-stage 엔진 실동작 테스트.
///
/// 컴파일된 `.mlmodelc` 7개와 보이스 디렉터리가 필요하므로 환경변수가 있을 때만 돈다:
///   KOKORO_7STAGE_MODELS=<dir with *.mlmodelc>
///   KOKORO_VOICES=<dir with *.bin>
@Suite("KokoroSevenStageEngine")
struct SevenStageEngineTests {
    private static var modelDir: URL? {
        ProcessInfo.processInfo.environment["KOKORO_7STAGE_MODELS"].map(URL.init(fileURLWithPath:))
    }
    private static var voiceDir: URL? {
        ProcessInfo.processInfo.environment["KOKORO_VOICES"].map(URL.init(fileURLWithPath:))
    }

    @Test("합성 + 단어 타임스탬프", .enabled(if: modelDir != nil && voiceDir != nil))
    func synthesizeWithTimestamps() throws {
        let engine = try KokoroSevenStageEngine(
            modelDirectory: Self.modelDir!, voicesDirectory: Self.voiceDir!
        )
        let text = "The quick brown fox jumps over the lazy dog."
        let result = try engine.synthesize(text: text, voice: "af_heart", speed: 1.0)

        print("phonemes: \(result.phonemes)")
        print(String(
            format: "samples=%d  duration=%.3fs  synth=%.1fms  RTF=%.1fx",
            result.samples.count, result.duration,
            result.synthesisTime * 1000, result.realTimeFactor
        ))
        print("timestamps (\(result.timestamps.count)):")
        for t in result.timestamps {
            print(String(format: "  %6.3f – %6.3f  %@", t.start, t.end, t.text))
        }

        #expect(!result.samples.isEmpty)
        #expect(result.duration > 1.0 && result.duration < 8.0)
        #expect(!result.timestamps.isEmpty)

        // 단조 증가 · 오디오 길이 안에 들어와야 한다.
        var previousEnd = -1.0
        for t in result.timestamps {
            #expect(t.start >= 0)
            #expect(t.end >= t.start)
            #expect(t.start >= previousEnd - 0.001, "타임스탬프가 역행: \(t)")
            previousEnd = t.end
        }
        #expect(previousEnd <= result.duration + 0.05, "마지막 타임스탬프가 오디오를 넘어감")

        // 마지막 단어의 끝이 오디오 끝 근처여야 한다(전체를 덮는지).
        #expect(previousEnd > result.duration * 0.7, "타임스탬프가 오디오 뒷부분을 못 덮음")

        // 감상용으로 저장.
        let out = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("sevenstage.wav")
        try writeWAV(result.samples, to: out)
        print("wav: \(out.path)")
    }

    /// 스테이지 입력 dtype 계약을 고정한다.
    ///
    /// 이 검사가 필요한 이유: macOS CoreML은 dtype 불일치를 관대하게 넘겨 주지만 iOS 기기는
    /// `Cannot retrieve vector from IRValue format …`으로 거부한다. 즉 **합성이 macOS에서
    /// 통과해도 기기에서 죽을 수 있다** — 실제로 Noise(fp32 출력) → Vocoder(fp16 입력) 경계에서
    /// 그렇게 당했다. 동작 테스트로는 못 잡으므로 계약을 직접 본다.
    @Test("스테이지 입력 dtype 계약", .enabled(if: modelDir != nil))
    func stageInputDataTypes() throws {
        let expected: [String: [String: MLMultiArrayDataType]] = [
            "KokoroAlbert": ["input_ids": .int32, "attention_mask": .int32],
            "KokoroPostAlbert": [
                "bert_dur": .float16, "input_ids": .int32, "style_s": .float16,
                "speed": .float16, "attention_mask": .int32,
            ],
            "KokoroAlignment": ["pred_dur": .int32, "d": .float16, "t_en": .float16],
            "KokoroProsody": ["en": .float16, "style_s": .float16],
            "KokoroNoise": ["F0_curve": .float32, "style_timbre": .float32],
            "KokoroVocoder": [
                "asr": .float16, "F0_curve": .float16, "N_pred": .float16,
                "x_source_0": .float16, "x_source_1": .float16, "style_timbre": .float16,
            ],
            "KokoroTail": ["x_pre": .float32],
        ]

        for (name, inputs) in expected.sorted(by: { $0.key < $1.key }) {
            let url = Self.modelDir!.appendingPathComponent("\(name).mlmodelc")
            let model = try MLModel(contentsOf: url)
            for (input, type) in inputs.sorted(by: { $0.key < $1.key }) {
                let desc = model.modelDescription.inputDescriptionsByName[input]
                #expect(desc != nil, "\(name): 입력 \(input) 없음")
                #expect(
                    desc?.multiArrayConstraint?.dataType == type,
                    "\(name).\(input): \(String(describing: desc?.multiArrayConstraint?.dataType)) ≠ \(type)"
                )
            }
        }
    }

    private func writeWAV(_ samples: [Float], to url: URL) throws {
        let sr = UInt32(KokoroSevenStageEngine.sampleRate)
        var pcm = [Int16]()
        pcm.reserveCapacity(samples.count)
        for s in samples { pcm.append(Int16(max(-1, min(1, s)) * 32767)) }
        let dataBytes = pcm.count * 2
        var d = Data()
        func le<T: FixedWidthInteger>(_ v: T) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
        d.append(contentsOf: Array("RIFF".utf8)); le(UInt32(36 + dataBytes))
        d.append(contentsOf: Array("WAVEfmt ".utf8)); le(UInt32(16)); le(UInt16(1)); le(UInt16(1))
        le(sr); le(sr * 2); le(UInt16(2)); le(UInt16(16))
        d.append(contentsOf: Array("data".utf8)); le(UInt32(dataBytes))
        pcm.withUnsafeBufferPointer { d.append(UnsafeRawBufferPointer($0).bindMemory(to: UInt8.self)) }
        try d.write(to: url)
    }
}
