import Foundation
import Testing
@testable import KokoroCoreML

/// 문서에 흔한 특수 표기의 읽기 회귀 테스트.
@Suite("특수 표기 읽기")
struct ReadingPatternTests {
    private let g2p = EnglishG2P(british: false)
    private func p(_ t: String) -> String { g2p.phonemize(text: t).0 }
    /// 두 음소 덩어리가 공백 없이 붙었는지(= 한 단어로 뭉개 읽는지).
    private func hasGap(_ s: String) -> Bool { s.contains(" ") }

    @Test("도메인은 dot으로 끊어 읽는다")
    func domains() {
        #expect(p("example.com").contains("dˈɑt"))
        #expect(p("www.gutenberg.org").contains("dˈɑt"))
        // www는 철자로, 도메인 몸통은 정상 발음으로.
        #expect(p("www.gutenberg.org").contains("dˌʌbᵊlju"))
        #expect(p("gutenberg.org").contains("ɡˈutᵊnbˌɜɹɡ"))
        // 도메인이 아닌 것은 건드리지 않는다.
        #expect(!p("file.txt").contains("dˈɑt"))
        #expect(!p("3.5").contains("dˈɑt"))
        #expect(!p("U.S.A.").contains("dˈɑt"))
    }

    @Test("큰 수")
    func bigNumbers() {
        #expect(p("1000000").contains("mˈɪljᵊn"))
        #expect(p("1,000,000").contains("mˈɪljᵊn"))
        #expect(p("1000000000").contains("bˈɪljən"))
        #expect(p("5000").contains("θˈWzᵊnd"))
    }

    @Test("숫자 범위는 to로 읽는다")
    func ranges() {
        #expect(p("5-10").contains("tə"))
        #expect(p("1\u{2013}3").contains("tə"))       // en dash
        #expect(p("1914-1918").contains("tə"))
        // 날짜·전화번호는 범위가 아니다.
        #expect(!p("2026-08-18").contains(" tə "))
        #expect(!p("555-1234").contains(" tə "))
    }

    @Test("쪽 표기")
    func pages() {
        #expect(p("p. 12").contains("pˈAʤ"))
        #expect(p("pp. 12-15").contains("pˈAʤᵻz"))
        // 숫자가 안 따라오면 건드리지 않는다.
        #expect(!p("a p. b").contains("pˈAʤ"))
    }

    @Test("슬래시는 양쪽을 띄운다")
    func slashes() {
        #expect(hasGap(p("and/or")))
        #expect(hasGap(p("he/she")))
    }

    @Test("숫자+단위에서 숫자가 사라지지 않는다")
    func numberUnit() {
        #expect(p("5kg").contains("fˈIv"))
        // 서수는 그대로.
        #expect(p("1st") == "fˈɜɹst")
        #expect(p("21st").contains("fˈɜɹst"))
    }

    @Test("vs. 는 versus")
    func versus() {
        #expect(p("vs.").contains("vˈɜɹsəs"))
    }
}
