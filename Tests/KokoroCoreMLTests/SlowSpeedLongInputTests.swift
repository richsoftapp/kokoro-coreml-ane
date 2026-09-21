import Foundation
import Testing

@testable import KokoroCoreML

/// 느린 배속 × 긴 그룹. duration ∝ 1/speed 라 510토큰 그룹이 Prosody/Vocoder의 shape range 상한
/// (2000프레임)을 넘기면 CoreML이 throw 해 청크 전체가 실패한다. 환경변수가 있을 때만 돈다.
@Suite("Slow speed × long input")
struct SlowSpeedLongInputTests {
    private static var modelDir: URL? {
        ProcessInfo.processInfo.environment["KOKORO_7STAGE_MODELS"].map(URL.init(fileURLWithPath:))
    }
    private static var voiceDir: URL? {
        ProcessInfo.processInfo.environment["KOKORO_VOICES"].map(URL.init(fileURLWithPath:))
    }

    static let texts: [(String, String)] = [
        ("plain", String(repeating: "The quick brown fox jumps over the lazy dog near the riverbank. ", count: 14)),
        ("numbers", String(repeating: "In 1994 the company earned 1,234,567 dollars and 89 cents on 365 days. ", count: 8)),
    ]

    @Test("speed 0.5에서 긴 입력이 throw 없이 합성된다", .enabled(if: modelDir != nil && voiceDir != nil))
    func slowSpeedLongInput() throws {
        let engine = try KokoroSevenStageEngine(modelDirectory: Self.modelDir!, voicesDirectory: Self.voiceDir!)
        for (label, text) in Self.texts {
            let (phonemes, _) = engine.g2pForTesting.phonemize(text: text)
            print("▶ \(label): \(text.count)자 → 음소 \(phonemes.count)자")
            for speed: Float in [1.0, 0.5] {
                do {
                    let r = try engine.synthesize(text: text, voice: "af_heart", speed: speed)
                    let frames = r.samples.count / KokoroSevenStageEngine.hopSize
                    print(String(format: "   speed %.2f ✓ %.1fs · %d frames · %.2f frames/phoneme-char",
                                 speed, r.duration, frames, Double(frames) / Double(max(1, phonemes.count))))
                    #expect(!r.samples.isEmpty)
                    // 상한을 넘겨 오디오만 잘리면 pred_dur로 계산한 타임스탬프가 오디오 끝을 넘어선다 —
                    // 그게 "뒷부분 단어가 소리 없이 빠진" 증거다. 수정 후엔 둘이 같은 pred_dur에서 나온다.
                    let lastEnd = r.timestamps.map(\.end).max() ?? 0
                    #expect(lastEnd <= r.duration + 0.1,
                            "\(label) @ \(speed): 마지막 타임스탬프 \(lastEnd)s > 오디오 \(r.duration)s — 그룹 뒷부분이 잘렸다")
                } catch {
                    print("   speed \(speed) ✗ throw: \(error)")
                    Issue.record("\(label) @ \(speed): \(error)")
                }
            }
        }
    }
}
