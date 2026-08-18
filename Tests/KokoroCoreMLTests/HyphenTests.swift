import Foundation
import Testing
@testable import KokoroCoreML

/// 하이픈 처리 회귀 테스트.
///
/// 단어를 잇는 하이픈은 **쉼이 아니다**. 예전엔 `—`(em dash)를 음소로 넣어 Kokoro가 그 자리에서
/// 멈췄고("well [쉼] known"), 게다가 쪼개진 조각이 홀로 사전 조회를 타 "re-use"의 "re"가
/// 음이름 ɹˌA("레이")로 읽혔다.
@Suite("하이픈")
struct HyphenTests {
    private let g2p = EnglishG2P(british: false)
    private func phonemes(_ text: String) -> String { g2p.phonemize(text: text).0 }

    @Test("단어 내부 하이픈은 쉼을 만들지 않는다")
    func noPauseInsideWords() {
        for word in ["well-known", "state-of-the-art", "twenty-five", "e-mail", "self-driving"] {
            let p = phonemes(word)
            #expect(!p.contains("—"), "\(word) → \(p) 에 쉼(—)이 들어갔다")
        }
    }

    @Test("사전에 있는 복합어는 붙여 쓴 발음을 쓴다")
    func lexiconCompoundsUseJoinedPronunciation() {
        #expect(phonemes("re-use") == phonemes("reuse"))
        #expect(!phonemes("re-use").contains("—"))
    }

    @Test("독립된 대시는 쉼으로 남는다")
    func standaloneDashKeepsPause() {
        #expect(phonemes("a - b").contains("—"))
        #expect(phonemes("a — b").contains("—"))
    }

    @Test("사전에 통째로 있는 하이픈 단어는 그대로")
    func hyphenatedLexiconEntriesUnchanged() {
        // "co-op"은 사전에 하이픈째 있으므로 "coop"(kˈup)으로 바뀌면 안 된다.
        let p = phonemes("co-op")
        #expect(p == "kˈOˌɑp", "co-op → \(p)")
    }
}
