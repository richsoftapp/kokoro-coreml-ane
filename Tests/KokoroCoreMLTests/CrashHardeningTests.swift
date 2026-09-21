import Foundation
import Testing

@testable import KokoroCoreML

/// 실제 문서 입력에서 프로세스를 죽이던 경로들의 회귀 테스트. 모델 없이 도는 순수 로직만 담는다.
@Suite("Crash hardening")
struct CrashHardeningTests {
    private let g2p = EnglishG2P(british: false)

    @Test("마크다운 링크 앞에 이모지·결합문자가 있어도 죽지 않고 라벨을 읽는다")
    func markdownLinkAfterWideCharacters() {
        // 예전엔 NSRange(UTF-16) 오프셋을 Character 오프셋으로 써서 "String index is out of bounds"로 트랩.
        let here = g2p.phonemize(text: "here").0
        let emoji = g2p.phonemize(text: "😀😀😀😀😀😀😀😀😀😀 [here](x)").0
        #expect(emoji.contains(here), "\(emoji)")
        let combining = g2p.phonemize(text: "cafe\u{0301} cafe\u{0301} cafe\u{0301} [here](x) [there](y)").0
        #expect(combining.contains(here), "\(combining)")
        // 사용자 음소 문법은 그대로 동작해야 한다(링크 앞에 이모지가 있어도).
        #expect(g2p.phonemize(text: "[word](/wɜɹd/)").0.contains("wɜɹd"))
        #expect(g2p.phonemize(text: "🙂 [word](/wɜɹd/)").0.contains("wɜɹd"))
    }

    @Test("2^63 같은 큰 소수는 abs(Int.min) 트랩 없이 자리로 읽는다")
    func hugeDecimalDoesNotTrap() {
        let n2w = EnglishNum2Word()
        // NSDecimalNumber.intValue가 정확히 Int.min을 돌려주는 값.
        let twoTo63 = Decimal(string: "9223372036854775808.5")!
        let spoken = n2w.convert(twoTo63)
        #expect(spoken.hasPrefix("nine two two"), "\(spoken)")
        #expect(n2w.convert(Decimal(string: "12345678901234567890123")!).hasPrefix("one two three"))
        #expect(n2w.convert(Decimal(string: "99999999999999999999")!, to: .year).hasPrefix("nine nine"))
        #expect(n2w.convert(Decimal(string: "99999999999999999999")!, to: .ordinal) == "")
        // Int 범위 안은 예전과 같다.
        #expect(n2w.convert(Decimal(2025)) == "two thousand, twenty-five")
        #expect(n2w.convert(Decimal(string: "3.14")!) == "three point one four")
        // G2P 끝까지 통과.
        for text in ["Balance: 9223372036854775808.5 units", "id 9223372036854775808.0", "-9223372036854775808.5"] {
            #expect(!g2p.phonemize(text: text).0.isEmpty, "\(text)")
        }
    }

    @Test("duration의 NaN·inf·과대값·합 초과를 방어한다")
    func clampDurations() {
        #expect(KokoroSevenStageEngine.clampDurations([.nan, .infinity, -.infinity, 0, 2.4, 2.5, 70000])
                == [1, 1, 1, 1, 2, 3, KokoroSevenStageEngine.maxFramesPerToken])

        // 합이 상한을 넘으면 비례 축소(최소 1 유지).
        let long = [Float](repeating: 10, count: 500)  // 5000
        let scaled = KokoroSevenStageEngine.clampDurations(long)
        #expect(scaled.count == 500)
        #expect(scaled.reduce(0, +) <= KokoroSevenStageEngine.maxFrames)
        #expect(scaled.allSatisfy { $0 >= 1 })

        // 바닥값 1로 올라간 항 때문에 아직 넘치는 경우도 끝내 상한 안으로.
        let mixed = [Float](repeating: 0.6, count: 400) + [Float](repeating: 60, count: 100)  // 400 + 6000
        let fixed = KokoroSevenStageEngine.clampDurations(mixed)
        #expect(fixed.count == 500)
        #expect(fixed.reduce(0, +) <= KokoroSevenStageEngine.maxFrames)
        #expect(fixed.allSatisfy { $0 >= 1 })

        // 상한 안이면 손대지 않는다.
        #expect(KokoroSevenStageEngine.clampDurations([3, 4, 5]) == [3, 4, 5])
    }

    @Test("속도 정규화와 속도별 토큰 예산")
    func speedBudget() {
        #expect(KokoroSevenStageEngine.sanitizedSpeed(0) == 1)
        #expect(KokoroSevenStageEngine.sanitizedSpeed(-1) == 1)
        #expect(KokoroSevenStageEngine.sanitizedSpeed(.nan) == 1)
        #expect(KokoroSevenStageEngine.sanitizedSpeed(.infinity) == 1)
        #expect(KokoroSevenStageEngine.sanitizedSpeed(0.01) == 0.1)
        #expect(KokoroSevenStageEngine.sanitizedSpeed(1.5) == 1.5)

        #expect(KokoroSevenStageEngine.tokenBudget(forSpeed: 2.0) == 510)
        #expect(KokoroSevenStageEngine.tokenBudget(forSpeed: 1.0) == 510)
        #expect(KokoroSevenStageEngine.tokenBudget(forSpeed: 0.8) == 510)
        #expect(KokoroSevenStageEngine.tokenBudget(forSpeed: 0.5) == 318)
        #expect(KokoroSevenStageEngine.tokenBudget(forSpeed: 0.1) == 64)
    }

    @Test("MLMultiArray 채우기는 값 개수가 shape와 다르면 throw 한다")
    func multiArrayFillChecksCount() throws {
        #expect(throws: KokoroError.self) { try MLArrays.int32([1, 2, 3], shape: [1, 2]) }
        #expect(throws: KokoroError.self) { try MLArrays.float32([1], shape: [1, 2]) }
        #expect(throws: KokoroError.self) { try MLArrays.float16([], shape: [0]) }
        let ok = try MLArrays.float16([1, 2], shape: [1, 2])
        #expect(ok.count == 2)
    }

    @Test("빈 alias 토큰이 사전 조회에서 죽지 않는다")
    func emptyAliasDoesNotTrap() {
        let lexicon = Lexicon(british: false)
        let text = "x"
        let token = MToken(text: text, tokenRange: text.startIndex..<text.endIndex, whitespace: "")
        token.meta.alias = ""
        let out = lexicon.transcribe(token, ctx: TokenContext())
        #expect(out.0 == nil)
    }
}
