import Foundation
import Testing

@testable import KokoroCoreML

/// Swift 포팅이 NLTK `PerceptronTagger`와 토큰 단위로 완전히 같은지 — 골든 파일은 NLTK 3.10이
/// Penn Treebank 샘플 300문장(7,095토큰)을 태깅한 결과다. 알고리즘이 결정적이므로 1개라도
/// 다르면 자질 이름·정규화·동점 규칙 중 무언가가 어긋난 것이다.
@Suite("PerceptronTagger")
struct PerceptronTaggerTests {
    @Test("NLTK 골든과 토큰 단위 일치")
    func matchesNLTKGolden() throws {
        let url = try #require(Bundle.module.url(forResource: "perceptron_golden", withExtension: "json"))
        let golden = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [[[String]]]
        let sentences = try #require(golden)
        var mismatches: [String] = []
        var total = 0
        for sentence in sentences {
            let words = sentence.map { $0[0] }
            let expected = sentence.map { $0[1] }
            let got = PerceptronTagger.shared.tag(words)
            for (i, word) in words.enumerated() where got[i] != expected[i] {
                mismatches.append("\(word): \(got[i]) ≠ \(expected[i])")
            }
            total += words.count
        }
        #expect(total > 7000)
        #expect(mismatches.isEmpty, "\(mismatches.count)/\(total) 불일치: \(mismatches.prefix(10))")
    }

    @Test("관사·기능어 태그")
    func functionWords() {
        let tags = PerceptronTagger.shared.tag(
            "turns any text into a natural-sounding voice .".components(separatedBy: " "))
        #expect(tags == ["VBZ", "DT", "NN", "IN", "DT", "JJ", "NN", "."])
    }
}
