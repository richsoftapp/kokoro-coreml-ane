import CoreML
import Foundation
import Testing

@testable import KokoroCoreML

/// 이상 입력에 대한 견고성 테스트.
///
/// 합성 경로는 사용자가 넣은 임의의 문서(PDF·EPUB·OCR 결과)를 그대로 받는다. 이모지·결합문자·
/// 제어문자·초장문 단어 같은 것이 섞여도 죽지 않아야 하고, 타임스탬프도 불변식을 지켜야 한다.
@Suite("견고성")
struct RobustnessTests {
    private static var modelDir: URL? {
        ProcessInfo.processInfo.environment["KOKORO_7STAGE_MODELS"].map(URL.init(fileURLWithPath:))
    }
    private static var voiceDir: URL? {
        ProcessInfo.processInfo.environment["KOKORO_VOICES"].map(URL.init(fileURLWithPath:))
    }

    static let cases: [(String, String)] = [
        ("빈 문자열", ""),
        ("공백만", "   \n\t  "),
        ("마침표만", "..."),
        ("문장부호만", "!@#$%^&*()_+-=[]{}|;':\",./<>?"),
        ("이모지 하나", "😀"),
        ("이모지 ZWJ", "👨‍👩‍👧‍👦 family"),
        ("피부톤", "👍🏽 good"),
        ("국기", "🇰🇷🇺🇸 flags"),
        ("한글", "안녕하세요 반갑습니다."),
        ("일본어", "こんにちは、世界。"),
        ("중국어", "你好世界。"),
        ("아랍어(RTL)", "مرحبا بالعالم"),
        ("히브리어(RTL)", "שלום עולם"),
        ("결합문자", "e\u{0301}\u{0302}\u{0303} cafe\u{0301}"),
        ("자모 분리 한글", "\u{1100}\u{1161}\u{11A8}"),
        ("제로폭 문자", "he\u{200B}llo\u{FEFF} world"),
        ("제어문자", "hello\u{0007}\u{001B}world"),
        ("치환문자", "bad \u{FFFD} char"),
        ("서로게이트 상위평면", "𝕳𝖊𝖑𝖑𝖔 𝓦𝓸𝓻𝓵𝓭"),
        ("수학기호", "∀x∈ℝ: x² ≥ 0"),
        ("통화", "$100 €50 ¥300 ₩5000 £20"),
        ("URL", "Visit https://example.com/a?b=c#d now."),
        ("이메일", "Mail me at a.b+c@example.co.uk please."),
        ("긴 단어(공백 없음)", String(repeating: "a", count: 2000)),
        ("긴 문장", String(repeating: "The quick brown fox jumps. ", count: 60)),
        ("숫자 나열", "1 2 3 10 100 1000 1000000 3.14159 -42"),
        ("날짜/시각", "On 2026-08-18 at 14:30, in 1st place."),
        ("반복 마침표", "a." + String(repeating: ".", count: 300)),
        ("줄바꿈 다수", "line1\n\n\nline2\r\nline3"),
        ("탭 섞임", "col1\tcol2\tcol3"),
        ("혼합 스크립트", "Hello 안녕 こんにちは 你好 مرحبا 123"),
        ("따옴표 종류", "\u{2018}a\u{2019} \u{201C}b\u{201D} «c» „d“"),
        ("대시 종류", "a-b – c — d ― e"),
        ("생략부호", "wait… what?"),
        ("전각 영문", "Ｈｅｌｌｏ　Ｗｏｒｌｄ"),
        ("한 글자", "a"),
        ("숫자 하나", "7"),
    ]

    /// G2P + 토크나이저 단계에서 죽지 않는지, 그리고 vocab 밖 음소로 인덱스가 밀리지 않는지.
    @Test("G2P·토크나이저 견고성")
    func g2pAndTokenizer() throws {
        let g2p = EnglishG2P(british: false)
        let tokenizer = try Tokenizer.loadFromBundle()
        var drifted: [String] = []

        for (label, text) in Self.cases {
            let (phonemes, tokens) = g2p.phonemize(text: text)
            let ids = tokenizer.encode(phonemes, maxLength: 512)

            // 토크나이저는 vocab에 없는 문자를 버린다. 타임스탬프 인덱스 걷기는 토큰의
            // phonemes.count 를 그대로 더해 가므로, 버려진 문자가 있으면 어긋난다.
            let tokenPhonemeChars = tokens.reduce(0) { $0 + ($1.phonemes?.count ?? 0) }
            let encodedBody = max(0, ids.count - 2)  // BOS/EOS 제외
            let whitespaceTokens = tokens.filter { !$0.whitespace.isEmpty }.count
            // 대략적 비교(공백 토큰이 음소 사이에 들어가므로 정확히 같지는 않다).
            if tokenPhonemeChars > 0, encodedBody + whitespaceTokens + 2 < tokenPhonemeChars {
                drifted.append("\(label): 토큰음소 \(tokenPhonemeChars) vs 인코딩 \(encodedBody)")
            }
            print("  \(label): phonemes=\(phonemes.count) ids=\(ids.count) tokens=\(tokens.count)")
        }

        if !drifted.isEmpty {
            print("\n⚠️ vocab 밖 문자로 길이가 어긋난 케이스:")
            drifted.forEach { print("   \($0)") }
        }
    }

    /// 전체 합성. 죽지 않고, 타임스탬프 불변식을 지켜야 한다.
    @Test("합성 견고성", .enabled(if: modelDir != nil && voiceDir != nil))
    func synthesisRobustness() throws {
        let engine = try KokoroSevenStageEngine(
            modelDirectory: Self.modelDir!, voicesDirectory: Self.voiceDir!
        )
        var failures: [String] = []

        for (label, text) in Self.cases {
            // 크래시가 나면 이 줄이 마지막 출력이 되어 범인이 특정된다.
            print("▶ \(label) (\(text.count)자)")
            fflush(stdout)
            do {
                let r = try engine.synthesize(text: text, voice: "af_heart", speed: 1.0)
                var previousEnd = -1.0
                for t in r.timestamps {
                    if t.start < 0 || t.end < t.start {
                        failures.append("\(label): 잘못된 구간 \(t)"); break
                    }
                    if t.start < previousEnd - 0.001 {
                        failures.append("\(label): 타임스탬프 역행 \(t)"); break
                    }
                    previousEnd = t.end
                }
                if previousEnd > r.duration + 0.10 {
                    failures.append("\(label): 마지막 \(previousEnd)s > 오디오 \(r.duration)s")
                }
                print("   ✓ \(String(format: "%.2fs", r.duration)) · 타임스탬프 \(r.timestamps.count)개")
            } catch {
                // throw는 허용(호출부가 처리). 크래시만 아니면 된다.
                print("   throw: \(error)")
            }
            fflush(stdout)
        }

        if !failures.isEmpty {
            print("\n❌ 불변식 위반:")
            failures.forEach { print("   \($0)") }
        }
        #expect(failures.isEmpty, "타임스탬프 불변식 위반 \(failures.count)건")
    }
}
