import Foundation
import Testing

@testable import KokoroCoreML

/// 관사 "a"가 글자 이름("에이", ˈA)으로 읽히던 회귀 방어.
///
/// 원인은 품사였다 — Apple NLTagger의 영어 품사 자산이 없는 기기에서는 모든 낱말이 `.otherWord`로
/// 와 `a`가 관사로 잡히지 않았고, macOS에서는 자산이 있어 데스크톱 테스트만으로는 잡히지 않았다.
/// 지금은 내장 [[PerceptronTagger]]가 태그를 붙이므로 기기와 무관하다. 이 테스트는 음소가 아니라
/// **토큰 단위**로 본다 — 다른 낱말 안의 /eɪ/(plˈAn 등)와 섞이지 않게.
@Suite("Article")
struct ArticleTests {
    private func articlePhonemes(_ text: String, british: Bool = false) -> [String] {
        let g2p = EnglishG2P(british: british)
        let (_, tokens) = g2p.phonemize(text: text)
        return tokens.filter { $0.text == "a" || $0.text == "A" }.compactMap { $0.phonemes }
    }

    @Test("관사 a는 ɐ로 읽는다")
    func articleIsSchwa() {
        let cases = [
            "a natural-sounding voice",
            "turns any text into a natural-sounding voice.",
            "Pick a voice you like above, choose a speed that feels comfortable.",
            "on a plane, on a subway",
            "A quiet reader turns any text into speech.",
        ]
        for text in cases {
            let found = articlePhonemes(text)
            #expect(!found.isEmpty, "관사 토큰을 찾지 못함: \(text)")
            for phoneme in found {
                #expect(phoneme == "ɐ", "\"\(text)\"의 관사가 \(phoneme) (기대: ɐ)")
            }
        }
    }

    @Test("영국 영어 보이스에서도 같다")
    func articleIsSchwaBritish() {
        for phoneme in articlePhonemes("a natural-sounding voice", british: true) {
            #expect(phoneme == "ɐ")
        }
    }

    /// 긴 산문 한 단락(앱 온보딩 Follow along 본문) — 관사는 모두 ɐ, 전치사 in/on은 강세 없이.
    /// 품사 자산이 없는 기기에서 이 단락의 "a"가 전부 "에이"로 읽히던 게 이 수정의 출발점이다.
    @Test("긴 산문 단락 전체")
    func fullParagraph() {
        let text = """
            Welcome to Local TTS — a quiet, private reader that turns any text into a natural-sounding voice. \
            Paste an article, drop in a file, or start a blank note, and the words will be read back to you \
            sentence by sentence while the current line lights up, so your eyes and ears stay together. \
            It's made for long reads, language practice, and those tired evenings when looking at a screen \
            for one more minute feels like too much. Every voice runs on-device, so your text never leaves \
            you, and you can keep listening on a plane, on a subway, or anywhere the connection isn't great. \
            Pick a voice you like above, choose a speed that feels comfortable later in Settings, and let \
            the page do the reading for you. Tap any sentence to jump there, or press play below to start \
            from the top — and notice how the words you're hearing right now move with the highlight.
            """
        let started = Date()
        let (_, tokens) = EnglishG2P(british: false).phonemize(text: text)
        let elapsed = Date().timeIntervalSince(started)
        let articles = tokens.filter { $0.text == "a" }
        #expect(articles.count == 9)
        #expect(articles.allSatisfy { $0.phonemes == "ɐ" }, "\(articles.map { $0.phonemes ?? "?" })")
        let unstressed = tokens.filter { ["in", "on"].contains($0.text) && $0.phonemes?.hasPrefix("ˈ") == true }
        #expect(unstressed.isEmpty, "\(unstressed.map { "\($0.text)=\($0.phonemes ?? "?")" })")
        print("PROBE| paragraph phonemize \(String(format: "%.0f", elapsed * 1000))ms, \(tokens.count) tokens")
    }

    @Test("명사 뒤 낱글자 A는 글자 이름")
    func letterAfterNounStaysLetter() {
        #expect(articlePhonemes("See Exhibit A for details.") == ["ˈA"])
        #expect(articlePhonemes("Plan A failed, so we tried Plan B.") == ["ˈA"])
        #expect(articlePhonemes("Vitamin A deficiency") == ["ˈA"])
        // 문장 처음·전치사 뒤의 대문자 A는 여전히 관사.
        #expect(articlePhonemes("Once Upon A Time") == ["ɐ"])
    }
}
