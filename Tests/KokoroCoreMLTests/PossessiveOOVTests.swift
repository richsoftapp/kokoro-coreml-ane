import Foundation
import Testing

@testable import KokoroCoreML

/// 사전에 없는 낱말(주로 고유명사) 뒤 소유격 "'s" 회귀 테스트.
///
/// camelSplitRegex는 문자 클래스만 매치해 아포스트로피를 건너뛴다 — "McGregor's"를 쪼개면
/// ["Mc", "Gregor", "s"]가 되고, 외톨이 "s"가 두문자 철자 경로로 빠져 글자 이름
/// "에스"(ˈɛs)로 읽혔다. 어간을 먼저 풀고 그 발음 끝소리에 맞는 /s/·/z/·/ɪz/를 붙여야 한다.
@Suite("사전에 없는 낱말의 소유격")
struct PossessiveOOVTests {
    private let g2p = EnglishG2P(british: false)
    private func phonemes(_ text: String) -> String { g2p.phonemize(text: text).0 }

    @Test("고유명사 뒤 소유격은 글자 이름이 아니라 /z/로 잇는다")
    func possessiveAfterUnknownProperNoun() {
        let mcgregors = phonemes("McGregor's")
        #expect(!mcgregors.contains(" ˈɛs"), "\(mcgregors)")
        #expect(mcgregors.hasSuffix("z"), "\(mcgregors)")

        let sentence = phonemes("Mr. McGregor's garden")
        #expect(!sentence.contains(" ˈɛs"), "\(sentence)")
    }

    @Test("이미 사전에 있는 낱말의 소유격은 회귀 없이 그대로다")
    func possessiveAfterKnownWordUnaffected() {
        // "cat's"는 stem_s 경로(사전에 있는 낱말)가 이미 처리한다 — fallback을 타지 않아야 한다.
        let cats = phonemes("the cat's toy")
        #expect(cats.contains("kˈæts"), "\(cats)")
        #expect(!cats.contains(" ˈɛs"), "\(cats)")
        #expect(!phonemes("Peter's jacket").contains(" ˈɛs"))
    }
}
