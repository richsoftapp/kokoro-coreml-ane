import Foundation
import Testing

@testable import KokoroCoreML

/// 대문자로 쓰인 이름·낱말 읽기. 표지·머리말·헌사("—HOWARD MARKS")처럼 책 PDF에 흔한 전대문자 줄이
/// 철자("H O W A R D")로 읽히지 않아야 한다. 두문자(IBM)는 계속 철자로 읽는다.
@Suite("Caps names")
struct CapsNameTests {
    private let g2p = EnglishG2P(british: false)

    private func phonemes(_ text: String) -> String { g2p.phonemize(text: text).0 }

    /// 철자 읽기의 흔적: 글자 이름 음소가 연달아 붙는다(ˌAʧ = "H", ˌɛm = "M").
    private func looksSpelledOut(_ ph: String) -> Bool {
        ph.contains("ˌAʧ") || ph.contains("ˌɛmˌA")
    }

    @Test("대문자 이름은 낱말로 읽는다")
    func upperCaseNames() {
        let howard = phonemes("—HOWARD MARKS")
        #expect(!looksSpelledOut(howard), "\(howard)")
        #expect(howard.hasSuffix("mˈɑɹks"), "\(howard)")          // silver 낱말 발음
        #expect(phonemes("MORGAN HOUSEL") == "mˈɔɹɡᵊn hˈWsᵊl")
        #expect(phonemes("JOHN SMITH") == "ʤˈɑn smˈɪθ")
        // 대소문자만 다른 같은 이름은 같은 발음이어야 한다(HOWARD → BART 낱말 발음 = Howard).
        #expect(phonemes("HOWARD MARKS") == phonemes("Howard Marks"), "\(phonemes("HOWARD MARKS")) vs \(phonemes("Howard Marks"))")
    }

    /// 철자 읽기 결과에서 강세 기호를 뺀 것 — 글자 이름 음소가 이어진다(M→ɛm, S→ɛs, B→bi, C→si …).
    private func unstressed(_ ph: String) -> String {
        ph.replacingOccurrences(of: "ˈ", with: "").replacingOccurrences(of: "ˌ", with: "")
    }

    @Test("모음 없는 5글자 이상 대문자 두문자는 여전히 철자로 읽는다")
    func vowelLessAcronymsAreSpelled() {
        let letterNames: [Character: String] = [
            "M": "ɛm", "S": "ɛs", "N": "ɛn", "B": "bi", "C": "si", "L": "ɛl", "G": "ʤi", "T": "ti",
            "Q": "kju", "H": "Aʧ", "P": "pi", "K": "kA", "D": "di", "F": "ɛf",
        ]
        for acronym in ["MSNBC", "LGBTQ", "HTTPS", "PBKDF"] {
            let expected = acronym.map { letterNames[$0]! }.joined()
            let actual = unstressed(phonemes(acronym))
            #expect(actual == expected, "\(acronym): \(phonemes(acronym))")
        }
    }

    @Test("두문자·낱말 대문자는 기존대로")
    func acronymsAndWords() {
        #expect(phonemes("IBM") == "ˌIbˌiˈɛm")
        #expect(phonemes("NASA") == "nˈæsə")
        #expect(phonemes("THE END") == "ði ˈɛnd")
        #expect(phonemes("TIMELESS LESSONS ON WEALTH") == "tˈImləs lˈɛsənz ˈɔn wˈɛlθ")
        #expect(phonemes("NEW YORK TIMES") == "nˈu jˈɔɹk tˈImz")
    }
}
