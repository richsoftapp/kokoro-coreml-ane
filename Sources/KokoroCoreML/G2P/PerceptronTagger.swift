import Compression
import Foundation

/// Penn Treebank 품사 태거 — NLTK `PerceptronTagger`(Matthew Honnibal의 greedy averaged
/// perceptron, Apache 2.0)를 그대로 포팅한 것. 가중치도 NLTK가 배포하는 사전 학습 모델
/// (`averaged_perceptron_tagger_eng`)을 raw-deflate로 묶어 번들한다.
///
/// 왜 Apple `NLTagger` 대신 이걸 쓰나: NLTagger의 영어 품사 자산은 **기기에 없을 수 있다**
/// (새 기기, OS 업데이트 직후, 저장공간 회수 뒤). 그러면 모든 낱말이 `.other`/`.otherWord`로
/// 와서 품사에 기대는 읽기가 통째로 무너진다 — 관사 "a"가 글자 이름 "에이"로, "$100"이
/// 숫자 뒤 통화로 안 읽히는 식. 태거를 내장하면 어느 기기에서든 같은 태그가 나오고, 태그 체계가
/// Python Misaki(spaCy Penn 태그)와 같아져 NLTag→Penn 근사 변환이 필요 없다.
///
/// 알고리즘은 NLTK와 1:1이어야 한다 — 자질 이름 문자열까지 같아야 번들 가중치를 그대로 쓴다.
/// [[Tests/KokoroCoreMLTests/PerceptronTaggerTests.swift]]가 NLTK 출력(골든 300문장)과 토큰
/// 단위로 동일한지 확인한다.
final class PerceptronTagger: @unchecked Sendable {
    /// 프로세스 전체에서 한 번만 로드한다(가중치 7만 5천 자질).
    static let shared = PerceptronTagger()

    private static let start = ["-START-", "-START2-"]
    private static let end = ["-END-", "-END2-"]

    private let weights: [String: [String: Double]]
    /// 모호하지 않은 낱말(훈련 코퍼스에서 97% 이상 한 태그) → 태그. 모델을 거치지 않는다.
    private let tagdict: [String: String]
    /// 동점일 때 알파벳순으로 안정 선택하기 위해 정렬해 둔다(NLTK와 같은 규칙).
    private let classes: [String]

    private init() {
        guard let url = Bundle.module.url(forResource: "perceptron_tagger_eng.json", withExtension: "deflate"),
              let compressed = try? Data(contentsOf: url),
              let json = Self.inflate(compressed),
              let object = try? JSONSerialization.jsonObject(with: json) as? [String: Any],
              let rawWeights = object["weights"] as? [String: [String: Double]],
              let tagdict = object["tagdict"] as? [String: String],
              let classes = object["classes"] as? [String]
        else {
            fatalError("PerceptronTagger: 번들 모델(perceptron_tagger_eng.json.deflate)을 읽지 못함")
        }
        self.weights = rawWeights
        self.tagdict = tagdict
        self.classes = classes.sorted()
    }

    /// raw deflate(zlib 헤더 없음) 해제. Python 쪽은 `zlib.compressobj(9, DEFLATED, -15)`로 만든다.
    private static func inflate(_ data: Data) -> Data? {
        let capacity = 8 * 1024 * 1024  // 원본 JSON ~5MB
        let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: capacity)
        defer { buffer.deallocate() }
        let size = data.withUnsafeBytes { raw -> Int in
            guard let base = raw.bindMemory(to: UInt8.self).baseAddress else { return 0 }
            return compression_decode_buffer(buffer, capacity, base, data.count, nil, COMPRESSION_ZLIB)
        }
        guard size > 0, size < capacity else { return nil }
        return Data(bytes: buffer, count: size)
    }

    /// 한 문장의 토큰(낱말·구두점)에 Penn 태그를 붙인다. NLTK `PerceptronTagger.tag`와 동일.
    func tag(_ tokens: [String]) -> [String] {
        var prev = Self.start[0]
        var prev2 = Self.start[1]
        let context = Self.start + tokens.map(Self.normalize) + Self.end
        var output: [String] = []
        output.reserveCapacity(tokens.count)
        for (i, word) in tokens.enumerated() {
            let tag: String
            if let known = tagdict[word] {
                tag = known
            } else {
                tag = predict(features(i, word: word, context: context, prev: prev, prev2: prev2))
            }
            output.append(tag)
            prev2 = prev
            prev = tag
        }
        return output
    }

    /// NLTK `normalize`: 소문자화, 4자리 숫자는 !YEAR, 숫자로 시작하면 !DIGITS, 하이픈 포함은 !HYPHEN.
    private static func normalize(_ word: String) -> String {
        if word.contains("-"), word.first != "-" { return "!HYPHEN" }
        if word.count == 4, word.allSatisfy(\.isASCIIDigit) { return "!YEAR" }
        if let first = word.first, first.isASCIIDigit { return "!DIGITS" }
        return word.lowercased()
    }

    /// NLTK `_get_features`와 같은 자질 이름을 만든다(공백으로 이어 붙인 문자열이 곧 가중치 키).
    private func features(_ index: Int, word: String, context: [String], prev: String, prev2: String)
        -> [String]
    {
        let i = index + Self.start.count
        func suffix(_ s: String) -> String { String(s.suffix(3)) }
        return [
            "bias",
            "i suffix \(suffix(word))",
            "i pref1 \(word.first.map(String.init) ?? "")",
            "i-1 tag \(prev)",
            "i-2 tag \(prev2)",
            "i tag+i-2 tag \(prev) \(prev2)",
            "i word \(context[i])",
            "i-1 tag+i word \(prev) \(context[i])",
            "i-1 word \(context[i - 1])",
            "i-1 suffix \(suffix(context[i - 1]))",
            "i-2 word \(context[i - 2])",
            "i+1 word \(context[i + 1])",
            "i+1 suffix \(suffix(context[i + 1]))",
            "i+2 word \(context[i + 2])",
        ]
    }

    /// 자질 가중치 합이 가장 큰 태그. 동점이면 알파벳순 뒤쪽(NLTK `max(key=(score, label))`).
    private func predict(_ features: [String]) -> String {
        var scores: [String: Double] = [:]
        for feature in features {
            guard let tagWeights = weights[feature] else { continue }
            for (label, weight) in tagWeights {
                scores[label, default: 0] += weight
            }
        }
        var best = classes[0]
        var bestScore = scores[best] ?? 0
        for label in classes.dropFirst() {
            let score = scores[label] ?? 0
            if score > bestScore || (score == bestScore && label > best) {
                best = label
                bestScore = score
            }
        }
        return best
    }
}

private extension Character {
    var isASCIIDigit: Bool { isASCII && isNumber }
}
