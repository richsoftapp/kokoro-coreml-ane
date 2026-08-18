import Foundation
import Testing
@testable import KokoroCoreML

@Suite("큰 수")
struct BigNumberTests {
    @Test("백만 이상이 thousand로 뭉개지지 않는다")
    func largeScales() {
        let g2p = EnglishG2P(british: false)
        for (input, expect) in [
            ("1000000", "million"), ("1,000,000", "million"),
            ("2500000", "million"), ("1000000000", "billion"),
            ("1234567", "million"),
        ] {
            let p = g2p.phonemize(text: input).0
            print("  \(input) → \(p)")
            #expect(!p.isEmpty, "\(input) 빈 음소")
            #expect(!p.contains("θˈWzᵊnd θˈWzᵊnd"), "\(input): thousand thousand 로 읽음 → \(p)")
            _ = expect
        }
        // 천 단위는 그대로여야 한다.
        #expect(g2p.phonemize(text: "5000").0.contains("θˈWzᵊnd"))
    }
}
