// swift-format-ignore-file
// Originally from MisakiSwift by mlalma, Apache License 2.0
// BART neural fallback replaced with CamelCase splitting + letter spelling.

import BARTG2P
import Foundation
import NaturalLanguage

final class EnglishG2P {
    private let british: Bool
    private let tagger: NLTagger
    private let lexicon: Lexicon
    private let unk: String
    private let bart: BARTG2P?

    static let punctuationTags: Set<NLTag> = Set([
        .openQuote, .closeQuote, .openParenthesis, .closeParenthesis, .punctuation,
        .sentenceTerminator, .otherPunctuation,
    ])
    static let punctuations: Set<Character> = Set(";:,.!?\u{2014}\u{2026}\u{201C}\u{201D}\u{201E}")

    static let punctuationTagPhonemes: [String: String] = [
        "``": String(UnicodeScalar(8220)!),
        "\"\"": String(UnicodeScalar(8221)!),
        "''": String(UnicodeScalar(8221)!),
    ]

    static let nonQuotePunctuations: Set<Character> = Set(
        punctuations.filter { !"\u{201C}\u{201D}\u{201E}".contains($0) })
    static let vowels: Set<Character> = Set("AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ")
    static let consonants: Set<Character> = Set("bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ")
    static let subTokenJunks: Set<Character> = Set("',-._\u{2018}\u{2019}/")
    static let stresses = "ˌˈ"
    static let primaryStress = stresses[stresses.index(stresses.startIndex, offsetBy: 1)]
    static let secondaryStress = stresses[stresses.index(stresses.startIndex, offsetBy: 0)]
    static let subtokenizeRegexPattern =
        // swiftlint:disable:next line_length
        #"^[''']+|\p{Lu}(?=\p{Lu}\p{Ll})|(?:^-)?(?:\d?[,.]?\d)+|[-_]+|[''']{2,}|\p{L}*?(?:[''']\p{L})*?\p{Ll}(?=\p{Lu})|\p{L}+(?:[''']\p{L})*|[^-_\p{L}'''\d]|[''']+$"#
    // swiftlint:disable force_try
    static let subtokenizeRegex = try! NSRegularExpression(
        pattern: EnglishG2P.subtokenizeRegexPattern, options: [])
    /// Regex for splitting CamelCase and compound words into sub-parts.
    private static let camelSplitRegex = try! NSRegularExpression(
        pattern: #"[A-Z]{2,}(?=[A-Z][a-z])|[A-Z]{2,}$|[A-Z][a-z]*|[a-z]+"#, options: [])
    private static let linkRegex = try! NSRegularExpression(
        pattern: #"\[([^\]]+)\]\(([^\)]*)\)"#, options: [])
    // swiftlint:enable force_try

    private static let dipthongs: Set<Character> = Set("AIOQWYʤʧ")

    struct PreprocessFeature {
        enum Value {
            case int(Int)
            case double(Double)
            case string(String)
        }

        let value: Value
        let tokenRange: Range<String.Index>
    }

    init(british: Bool = false, unk: String = "❓") {
        self.british = british
        self.tagger = NLTagger(tagSchemes: [.nameTypeOrLexicalClass])
        self.lexicon = Lexicon(british: british)
        self.unk = unk

        self.bart = BARTG2P.fromBundle()
    }

    private func tokenContext(_ ctx: TokenContext, ps: String?, token: MToken) -> TokenContext {
        var vowel = ctx.futureVowel

        if let ps = ps {
            for c in ps {
                if EnglishG2P.nonQuotePunctuations.contains(c) {
                    vowel = nil
                    break
                }

                if EnglishG2P.vowels.contains(c) {
                    vowel = true
                    break
                }

                if EnglishG2P.consonants.contains(c) {
                    vowel = false
                    break
                }
            }
        }
        let futureTo =
            (token.text == "to" || token.text == "To")
            || (token.text == "TO" && (token.tag == .particle || token.tag == .preposition))
        return TokenContext(futureVowel: vowel, futureTo: futureTo)
    }

    func stressWeight(_ phonemes: String?) -> Int {
        guard let phonemes else { return 0 }
        return phonemes.reduce(0) { sum, character in
            sum + (Self.dipthongs.contains(character) ? 2 : 1)
        }
    }

    private func resolveTokens(_ tokens: inout [MToken]) {
        let text =
            tokens.dropLast().map { $0.text + $0.whitespace }.joined()
            + (tokens.last?.text ?? "")
        let prespace =
            text.contains(" ") || text.contains("/")
            || Set(
                text.compactMap { c -> Int? in
                    if EnglishG2P.subTokenJunks.contains(c) { return nil }

                    if c.isLetter { return 0 }
                    if c.isNumber { return 1 }
                    return 2
                }
            ).count > 1

        for i in 0..<tokens.count {
            if tokens[i].phonemes == nil {
                if i == tokens.count - 1, let last = tokens[i].text.last,
                    EnglishG2P.nonQuotePunctuations.contains(last)
                {
                    tokens[i].phonemes = tokens[i].text
                    tokens[i].meta.rating = 3
                } else if tokens[i].text.allSatisfy({
                    EnglishG2P.subTokenJunks.contains($0)
                }) {
                    tokens[i].phonemes = nil
                    tokens[i].meta.rating = 3
                }
            } else if i > 0 {
                tokens[i].meta.prespace = prespace
            }
        }

        guard !prespace else { return }

        var indices: [(Bool, Int, Int)] = []
        for (i, tk) in tokens.enumerated() {
            if let ps = tk.phonemes, !ps.isEmpty {
                indices.append((ps.contains(Lexicon.primaryStress), stressWeight(ps), i))
            }
        }
        if indices.count == 2, tokens[indices[0].2].text.count == 1 {
            let i = indices[1].2
            tokens[i].phonemes = Lexicon.applyStress(tokens[i].phonemes, stress: -0.5)
            return
        } else if indices.count < 2
            || indices.map({ $0.0 ? 1 : 0 }).reduce(0, +) <= (indices.count + 1) / 2
        {
            return
        }
        indices.sort { ($0.0 ? 1 : 0, $0.1) < ($1.0 ? 1 : 0, $1.1) }
        let cut = indices.prefix(indices.count / 2)

        for x in cut {
            let i = x.2
            tokens[i].phonemes = Lexicon.applyStress(tokens[i].phonemes, stress: -0.5)
        }
    }

    typealias PreprocessTuple = (text: String, tokens: [String], features: [PreprocessFeature])

    private func preprocess(text: String) -> PreprocessTuple {
        var result = ""
        var tokens: [String] = []
        var features: [PreprocessFeature] = []

        var input = text.trimmingCharacters(in: .whitespacesAndNewlines)
        // Expand symbols that NLTagger swallows when attached to numbers
        for (sym, word) in [("%", " percent"), ("&", " and"), ("+", " plus"), ("@", " at")] {
            input = input.replacingOccurrences(of: sym, with: word)
        }
        var lastEnd = input.startIndex
        let ns = input as NSString
        let fullRange = NSRange(location: 0, length: ns.length)

        Self.linkRegex.enumerateMatches(in: input, options: [], range: fullRange) { match, _, _ in
            guard let m = match else { return }

            // NSRange는 UTF-16 오프셋이다. 예전엔 이를 `String.index(_:offsetBy:)`의 **Character** 오프셋으로
            // 썼는데, 링크 앞에 이모지·결합문자처럼 UTF-16 2단위 이상인 글자가 있으면 오프셋이 글자 수를
            // 넘어 "String index is out of bounds"로 죽었다(예: "😀😀😀😀😀😀😀😀😀😀 [here](x)"). 정식 변환을
            // 쓰고, 그래핌 경계에 안 걸리면 그 링크는 그냥 본문으로 둔다.
            guard let swiftRange = Range(m.range, in: input), swiftRange.lowerBound >= lastEnd else { return }
            let start = swiftRange.lowerBound
            let end = swiftRange.upperBound

            result += String(input[lastEnd..<start])
            tokens.append(
                contentsOf: String(input[lastEnd..<start]).split(separator: " ").map(
                    String.init))

            let grapheme = ns.substring(with: m.range(at: 1))
            let phoneme = ns.substring(with: m.range(at: 2))

            let tokenStartIndex = result.endIndex
            result += grapheme
            let tokenRange = tokenStartIndex..<result.endIndex

            if let intValue = Int(phoneme) {
                features.append(
                    PreprocessFeature(value: .int(intValue), tokenRange: tokenRange))
            } else if ["0.5", "+0.5"].contains(phoneme) {
                features.append(
                    PreprocessFeature(value: .double(0.5), tokenRange: tokenRange))
            } else if phoneme == "-0.5" {
                features.append(
                    PreprocessFeature(value: .double(-0.5), tokenRange: tokenRange))
            } else if phoneme.count > 1 && phoneme.first == "/" && phoneme.last == "/" {
                features.append(
                    PreprocessFeature(
                        value: .string(String(phoneme.dropLast())), tokenRange: tokenRange))
            } else if phoneme.count > 1 && phoneme.first == "#" && phoneme.last == "#" {
                features.append(
                    PreprocessFeature(
                        value: .string(String(phoneme.dropLast())), tokenRange: tokenRange))
            }

            tokens.append(grapheme)
            lastEnd = end
        }

        if lastEnd < input.endIndex {
            result += String(input[lastEnd...])
            tokens.append(
                contentsOf: String(input[lastEnd...]).split(separator: " ").map(String.init))
        }

        return (text: result, tokens: tokens, features: features)
    }

    private func tokenize(preprocessedText: PreprocessTuple) -> [MToken] {
        var mutableTokens: [MToken] = []

        tagger.string = preprocessedText.text
        tagger.setLanguage(
            .english,
            range: preprocessedText.text.startIndex..<preprocessedText.text.endIndex)
        let options: NLTagger.Options = []
        tagger.enumerateTags(
            in: preprocessedText.text.startIndex..<preprocessedText.text.endIndex,
            unit: .word,
            scheme: .nameTypeOrLexicalClass,
            options: options
        ) { tag, tokenRange in
            if let tag = tag {
                let word = String(preprocessedText.text[tokenRange])
                if tag == .whitespace, let lastToken = mutableTokens.last {
                    lastToken.whitespace = word
                } else {
                    mutableTokens.append(
                        MToken(
                            text: word, tokenRange: tokenRange, tag: tag, whitespace: ""))
                }
            }

            return true
        }

        for feature in preprocessedText.features {
            for token in mutableTokens {
                if token.tokenRange.contains(feature.tokenRange)
                    || feature.tokenRange.contains(token.tokenRange)
                {
                    switch feature.value {
                    case .int(let int):
                        token.meta.stress = Double(int)
                    case .double(let double):
                        token.meta.stress = double
                    case .string(let string):
                        if string.hasPrefix("/") {
                            token.meta.is_head = true
                            token.phonemes = String(string.dropFirst())
                            token.meta.rating = 5
                        } else if string.hasPrefix("#") {
                            token.meta.num_flags = String(string.dropFirst())
                        }
                    }
                }
            }
        }

        return mutableTokens
    }

    func mergeTokens(_ tokens: [MToken], unk: String? = nil) -> MToken {
        let stressSet = Set(tokens.compactMap { $0.meta.stress })
        let currencySet = Set(tokens.compactMap { $0.meta.currency })
        let ratings: Set<Int?> = Set(tokens.map { $0.meta.rating })

        var phonemes: String? = nil
        if let unk {
            var phonemeBuilder = ""
            for token in tokens {
                if token.meta.prespace,
                    !phonemeBuilder.isEmpty,
                    !(phonemeBuilder.last?.isWhitespace ?? false),
                    token.phonemes?.isEmpty == false
                {
                    phonemeBuilder += " "
                }
                phonemeBuilder += token.phonemes ?? unk
            }
            phonemes = phonemeBuilder
        }

        let mergedText =
            tokens.dropLast().map { $0.text + $0.whitespace }.joined()
            + (tokens.last?.text ?? "")

        func score(_ t: MToken) -> Int {
            return t.text.reduce(0) {
                $0 + (String($1) == String($1).lowercased() ? 1 : 2)
            }
        }
        let tagSource = tokens.max(by: { score($0) < score($1) })

        let tokenRangeStart = tokens.first!.tokenRange.lowerBound
        let tokenRangeEnd = tokens.last!.tokenRange.upperBound
        let flagChars = Set(tokens.flatMap { Array($0.meta.num_flags) })

        return MToken(
            text: mergedText,
            tokenRange: Range<String.Index>(
                uncheckedBounds: (lower: tokenRangeStart, upper: tokenRangeEnd)),
            tag: tagSource?.tag,
            whitespace: tokens.last?.whitespace ?? "",
            phonemes: phonemes,
            start_ts: tokens.first?.start_ts,
            end_ts: tokens.last?.end_ts,
            underscore: Underscore(
                is_head: tokens.first?.meta.is_head ?? false,
                alias: nil,
                stress: (stressSet.count == 1 ? stressSet.first : nil),
                currency: currencySet.max(),
                num_flags: String(flagChars.sorted()),
                prespace: tokens.first?.meta.prespace ?? false,
                rating: ratings.contains(where: { $0 == nil })
                    ? nil : ratings.compactMap { $0 }.min()
            )
        )
    }

    func foldLeft(_ tokens: [MToken]) -> [MToken] {
        var result: [MToken] = []
        for token in tokens {
            if let last = result.last, !token.meta.is_head {
                _ = result.popLast()
                let merged = mergeTokens([last, token], unk: unk)
                result.append(merged)
            } else {
                result.append(token)
            }
        }
        return result
    }

    /// `[낱말, "-", 낱말]`이 공백 없이 붙어 있고 **붙여 쓴 형태가 사전에 있으면** 하나로 합친다.
    ///
    /// `tokenize`는 "re-use"를 세 토큰으로 쪼개는데, 그러면 "re"가 홀로 남아 음이름 ɹˌA("레이")로
    /// 읽힌다. 붙인 "reuse"는 사전에 ɹijˈuz로 있으므로 그걸 쓴다. 사전에 없으면(well-known,
    /// twenty-five) 손대지 않고 조각별 처리에 맡긴다 — 그쪽은 하이픈의 쉼만 제거되면 정확하다.
    /// 숫자 뒤에 올 때만 풀어 읽는 단위. (단수형, 복수형)
    ///
    /// 한 글자 단위(`m` `s` `g` `h` `in`)는 **일부러 뺐다** — "born in 1990 in Seoul"의 `in`이
    /// "1990 inches"가 되는 식의 오탐이 실제로 생긴다. 여러 글자 약어만 다룬다.
    static let unitWords: [String: (String, String)] = [
        "kg": ("kilogram", "kilograms"), "mg": ("milligram", "milligrams"),
        "km": ("kilometer", "kilometers"), "cm": ("centimeter", "centimeters"),
        "mm": ("millimeter", "millimeters"), "ml": ("milliliter", "milliliters"),
        "kb": ("kilobyte", "kilobytes"), "mb": ("megabyte", "megabytes"),
        "gb": ("gigabyte", "gigabytes"), "tb": ("terabyte", "terabytes"),
        "ft": ("foot", "feet"), "lb": ("pound", "pounds"), "lbs": ("pound", "pounds"),
        "oz": ("ounce", "ounces"), "mi": ("mile", "miles"), "yd": ("yard", "yards"),
        "hr": ("hour", "hours"), "hrs": ("hour", "hours"),
        "min": ("minute", "minutes"), "mins": ("minute", "minutes"),
        "sec": ("second", "seconds"), "secs": ("second", "seconds"),
        "ms": ("millisecond", "milliseconds"), "mph": ("mile per hour", "miles per hour"),
        "kw": ("kilowatt", "kilowatts"), "mhz": ("megahertz", "megahertz"),
        "ghz": ("gigahertz", "gigahertz"),
    ]

    /// 숫자를 삼켜 버리던 기호들. 값 없이 alias 로 읽는다.
    static let symbolWords: [String: String] = [
        "=": "equals", "°": "degrees", "§": "section", "×": "times",
        // ¥ ₩ 는 여기 두지 않는다 — 통화 경로(Lexicon.currencies)가 "300 yen"처럼
        // 숫자 뒤로 어순을 바로잡아 주기 때문이다. 여기 두면 "yen 300"이 된다.
    ]

    /// 여러 낱말로 읽어야 하는 기호. alias 는 한 낱말만 받으므로 음소를 직접 이어 붙인다.
    static let symbolPhrases: [String: [String]] = [
        "±": ["plus", "or", "minus"],
        "÷": ["divided", "by"],
        "≈": ["approximately"],
        "≤": ["less", "than", "or", "equal", "to"],
        "≥": ["greater", "than", "or", "equal", "to"],
    ]

    /// 철자로 읽어야 자연스러운 두문자·확장자.
    static let spelledOut: Set<String> = [
        "phd", "msc", "bsc", "mba", "pdf", "png", "jpg", "jpeg", "gif", "svg",
        "csv", "html", "css", "url", "usb", "gps", "pdfs",
    ]

    /// 문서에 흔한 특수 표기를 읽을 수 있게 손본다. **토큰 텍스트는 바꾸지 않고** alias/공백만
    /// 조정하므로, 화면 단어와 타이밍 토큰의 글자 흐름이 그대로 유지된다(하이라이트 정렬 보존).
    ///
    ///  · 숫자 범위 `5-10` → "five **to** ten" (예전엔 "fiveten"으로 붙었다)
    ///  · 슬래시 `and/or`, `km/h` → 양쪽을 띄운다 (예전엔 "andor"로 붙었다)
    ///  · `p. 12` → "page 12",  `pp. 12-15` → "pages 12 to 15"
    ///  · `vs.` → "versus" (예전엔 "viz")
    func readSpecialPatterns(_ rawTokens: [MToken]) -> [MToken] {
        let tokens = splitNumberUnitTokens(rawTokens)
        guard !tokens.isEmpty else { return tokens }
        func isDigits(_ t: MToken) -> Bool {
            !t.text.isEmpty && t.text.allSatisfy(\.isNumber)
        }
        for (i, token) in tokens.enumerated() {
            guard token.phonemes == nil, token.meta.alias == nil else { continue }
            let text = token.text
            let lower = text.lowercased()

            // 여러 낱말 기호(± ÷ ≈ ≤ ≥)
            if let phrase = Self.symbolPhrases[text] {
                let parts = phrase.compactMap { lexicon.phonemesForWord($0) }
                if parts.count == phrase.count {
                    token.phonemes = parts.joined(separator: " ")
                    token.whitespace = " "
                    if i > 0 { tokens[i - 1].whitespace = " " }
                    continue
                }
            }

            // 숫자를 삼키던 기호: = ° § ×
            if let symbol = Self.symbolWords[text] {
                token.meta.alias = symbol
                token.whitespace = " "
                if i > 0 { tokens[i - 1].whitespace = " " }
                continue
            }

            // 철자로 읽을 두문자·확장자 (PhD → P H D, pdf → P D F)
            if Self.spelledOut.contains(lower), let letters = lexicon.getNNP(text).phoneme {
                token.phonemes = letters
                continue
            }

            // @ 는 앞말과 붙어 있어도 "at" 으로 읽는다.
            if text == "@" {
                token.meta.alias = "at"
                token.whitespace = " "
                if i > 0 { tokens[i - 1].whitespace = " " }
                continue
            }

            // 단위: 바로 앞이 숫자일 때만 풀어 읽는다.
            if let (singular, plural) = Self.unitWords[lower], i > 0, isDigits(tokens[i - 1]) {
                token.meta.alias = tokens[i - 1].text == "1" ? singular : plural
                token.whitespace = token.whitespace.isEmpty ? " " : token.whitespace
                tokens[i - 1].whitespace = " "
                continue
            }

            // 단위 사이의 슬래시는 "per" (km/h → kilometers per hour)
            if text == "/", i > 0, i + 1 < tokens.count,
                Self.unitWords[tokens[i - 1].text.lowercased()] != nil
                    || tokens[i - 1].meta.alias?.hasSuffix("s") == true,
                Self.unitWords[tokens[i + 1].text.lowercased()] != nil
                    || tokens[i + 1].text.lowercased() == "h"
            {
                token.meta.alias = "per"
                token.phonemes = nil
                token.whitespace = " "
                tokens[i - 1].whitespace = " "
                if tokens[i + 1].text.lowercased() == "h" { tokens[i + 1].meta.alias = "hour" }
                continue
            }

            // URL 스킴은 읽지 않는다. "https://example.com"을 그대로 읽으면
            // "t: example dot com"처럼 깨진다(https가 "t"로 뭉개짐). 읽는 앱에서는
            // 스킴을 빼고 호스트만 읽는 편이 자연스럽다.
            if ["http", "https", "ftp", "ftps", "mailto"].contains(lower),
                i + 1 < tokens.count, tokens[i + 1].text == ":"
            {
                token.phonemes = ""
                token.whitespace = ""
                var next = i + 1
                // ":" 와 뒤따르는 "/" 들을 함께 지운다.
                while next < tokens.count, [":", "/"].contains(tokens[next].text) {
                    tokens[next].phonemes = ""
                    tokens[next].meta.alias = nil
                    tokens[next].whitespace = ""
                    next += 1
                }
                continue
            }

            // vs / vs. → versus
            if lower == "vs" {
                token.meta.alias = "versus"
                continue
            }

            // p. / pp. + 숫자 → page(s). 뒤에 숫자가 와야만 적용한다("p"로 끝나는 문장 보호).
            // 토크나이저가 "pp."를 점까지 한 토큰으로 주기도 하고 "pp" + "."로 쪼개기도 한다.
            let bare = lower.hasSuffix(".") ? String(lower.dropLast()) : lower
            if bare == "p" || bare == "pp" {
                let dotIsSeparate = lower == bare
                let numberIndex = dotIsSeparate ? i + 2 : i + 1
                if (!dotIsSeparate || (i + 1 < tokens.count && tokens[i + 1].text == ".")),
                    numberIndex < tokens.count, isDigits(tokens[numberIndex])
                {
                    token.meta.alias = bare == "p" ? "page" : "pages"
                    if dotIsSeparate {
                        let dot = tokens[i + 1]
                        dot.meta.alias = ""
                        dot.phonemes = ""
                        dot.whitespace = " "
                    } else {
                        token.whitespace = " "
                    }
                    continue
                }
            }

            // 슬래시: 음소 없이 양쪽만 띄운다. "and slash or"보다 "and or"가 자연스럽다.
            if text == "/" {
                token.phonemes = ""
                token.whitespace = " "
                if i > 0 { tokens[i - 1].whitespace = " " }
                continue
            }

            // 숫자 범위: 숫자-숫자가 공백 없이 붙어 있을 때만. 앞뒤로 대시가 더 있으면
            // 날짜(2026-08-18)·ISBN이므로 건드리지 않는다. 전화번호는 오탐 가능성이 있으나
            // 책·기사에서는 쪽수·연도 범위가 압도적으로 흔하다.
            if text == "-" || text == "\u{2013}", i > 0, i + 1 < tokens.count,
                isDigits(tokens[i - 1]), isDigits(tokens[i + 1]),
                tokens[i - 1].whitespace.isEmpty, token.whitespace.isEmpty,
                !(i >= 2 && ["-", "\u{2013}"].contains(tokens[i - 2].text)),
                !(i + 2 < tokens.count && ["-", "\u{2013}"].contains(tokens[i + 2].text)),
                // 555-1234 같은 전화번호 모양(3자리-4자리)은 범위가 아니다.
                !(tokens[i - 1].text.count == 3 && tokens[i + 1].text.count == 4)
            {
                token.meta.alias = "to"
                token.whitespace = " "
                tokens[i - 1].whitespace = " "
            }
        }
        return tokens
    }

    /// `5kg`처럼 숫자 뒤에 단위가 붙은 토큰을 숫자와 단위로 나눈다.
    ///
    /// 예전엔 통째로 사전을 못 찾아 "keg"로 읽히며 **숫자 5가 통째로 사라졌다**. 서수
    /// (`1st` `2nd` `3rd` `4th`)는 이미 정상 처리되므로 건드리지 않는다.
    func splitNumberUnitTokens(_ tokens: [MToken]) -> [MToken] {
        var out: [MToken] = []
        for token in tokens {
            guard token.phonemes == nil, token.meta.alias == nil,
                let boundary = token.text.firstIndex(where: { !$0.isNumber }),
                boundary != token.text.startIndex,
                token.text[boundary...].allSatisfy(\.isLetter),
                !["st", "nd", "rd", "th"].contains(String(token.text[boundary...]).lowercased())
            else {
                out.append(token)
                continue
            }
            let number = MToken(copying: token)
            number.text = String(token.text[..<boundary])
            number.whitespace = " "
            number.meta.is_head = true
            out.append(number)

            let unit = MToken(copying: token)
            unit.text = String(token.text[boundary...])
            unit.whitespace = token.whitespace
            unit.meta.is_head = false
            out.append(unit)
        }
        return out
    }

    /// 도메인/호스트 토큰을 조각과 "dot"으로 펼친다.
    ///
    /// `tokenize`는 "www.gutenberg.org"를 한 토큰으로 남기는데, 그러면 점이 통째로 사라져
    /// "example.com"이 "example컴"(ɪɡzˈæmpəlkˌɑm)으로 붙고 `www`는 한 단어로 뭉개져
    /// "유-어-어"(jˈuɔɔ)로 읽힌다.
    ///
    /// 조각을 각각 독립 토큰으로 만들어 **일반 G2P 경로**(사전 → 폴백)를 그대로 타게 하고,
    /// 사이에 alias `"dot"` 토큰을 끼운다. 사전에 없는 조각(gutenberg)도 정상 발음된다 —
    /// 사전 안에서 처리하면 폴백을 못 타 철자로 읽히는 문제가 있었다.
    /// `www`만 예외로 철자 발음을 직접 넣는다.
    ///
    /// 마지막 조각이 알려진 TLD이거나 첫 조각이 `www`일 때만 펼친다 — `file.txt`·`U.S.A.`·
    /// `3.5`는 건드리지 않는다.
    func expandDomainTokens(_ tokens: [MToken]) -> [MToken] {
        var out: [MToken] = []
        for token in tokens {
            guard token.phonemes == nil, token.meta.alias == nil,
                let parts = Self.domainParts(token.text)
            else {
                out.append(token)
                continue
            }
            for (index, part) in parts.enumerated() {
                if index > 0 {
                    let dot = MToken(copying: token)
                    dot.text = "."
                    // 조각 사이를 띄운다 — 안 띄우면 음소가 한 덩어리로 붙어 "더블유더블유
                    // 더블유닷구텐베르크닷오르그"처럼 몰아쳐 읽는다.
                    dot.whitespace = " "
                    dot.phonemes = nil
                    dot.meta.alias = "dot"
                    dot.meta.is_head = false
                    out.append(dot)
                }
                let piece = MToken(copying: token)
                piece.text = part
                piece.whitespace = " "
                let spellIt = part.lowercased() == "www"
                    || (index == parts.count - 1 && Self.fileExtensions.contains(part.lowercased()))
                piece.phonemes = spellIt ? lexicon.getNNP(part).phoneme : nil
                piece.meta.is_head = (index == 0)
                out.append(piece)
            }
            out.last?.whitespace = token.whitespace
        }
        return out
    }

    /// 도메인처럼 취급할 파일 확장자. 철자로 읽어야 자연스러운 것들만 넣는다
    /// (`file.txt`는 "file text"로도 무난해 뺐다).
    static let fileExtensions: Set<String> = [
        "pdf", "png", "jpg", "jpeg", "gif", "svg", "csv", "html", "htm",
        "zip", "docx", "xlsx", "pptx", "mp3", "mp4", "json", "xml",
    ]

    /// 도메인처럼 보이면 점으로 나눈 조각을, 아니면 nil.
    static func domainParts(_ text: String) -> [String]? {
        let stripped = text.hasSuffix(".") ? String(text.dropLast()) : text
        guard stripped.contains("."), !stripped.hasPrefix(".") else { return nil }
        let parts = stripped.split(separator: ".").map(String.init)
        guard parts.count >= 2,
            parts.allSatisfy({ !$0.isEmpty && $0.allSatisfy { $0.isLetter || $0.isNumber } }),
            parts.contains(where: { $0.contains(where: \.isLetter) }),
            let tld = parts.last?.lowercased(),
            Lexicon.knownTLDs.contains(tld) || fileExtensions.contains(tld)
                || parts[0].lowercased() == "www"
        else { return nil }
        return parts
    }

    func joinHyphenatedCompounds(_ tokens: [MToken]) -> [MToken] {
        guard tokens.count >= 3 else { return tokens }
        var out: [MToken] = []
        var i = 0
        while i < tokens.count {
            let dash = i + 2 < tokens.count ? tokens[i + 1] : nil
            if let dash, dash.text == "-",
                tokens[i].whitespace.isEmpty, dash.whitespace.isEmpty,
                tokens[i].phonemes == nil, tokens[i + 2].phonemes == nil,
                tokens[i].meta.alias == nil, tokens[i + 2].meta.alias == nil,
                !tokens[i].text.isEmpty, tokens[i].text.allSatisfy(\.isLetter),
                !tokens[i + 2].text.isEmpty, tokens[i + 2].text.allSatisfy(\.isLetter),
                let joined = lexicon.phonemesForWord(tokens[i].text + tokens[i + 2].text)
            {
                let merged = mergeTokens(Array(tokens[i...(i + 2)]))
                merged.phonemes = joined
                merged.meta.rating = 4
                out.append(merged)
                i += 3
                continue
            }
            out.append(tokens[i])
            i += 1
        }
        return out
    }

    func subtokenize(word: String) -> [String] {
        let nsString = word as NSString
        let range = NSRange(location: 0, length: nsString.length)
        let matches = EnglishG2P.subtokenizeRegex.matches(in: word, options: [], range: range)

        return matches.map { match in
            nsString.substring(with: match.range)
        }
    }

    enum RetokenizedItem {
        case single(MToken)
        case compound([MToken])
    }

    // swiftlint:disable:next function_body_length
    func retokenize(_ rawTokens: [MToken]) -> [RetokenizedItem] {
        let tokens = joinHyphenatedCompounds(expandDomainTokens(readSpecialPatterns(rawTokens)))
        var words: [RetokenizedItem] = []
        var currency: String? = nil

        for (i, token) in tokens.enumerated() {
            let needsSplit = (token.meta.alias == nil && token.phonemes == nil)
            var subtokens: [MToken] = []
            if needsSplit {
                let parts = subtokenize(word: token.text)
                subtokens = parts.map { part in
                    let t = MToken(copying: token)
                    t.text = part
                    t.whitespace = ""
                    t.meta.is_head = true
                    t.meta.prespace = false
                    return t
                }
            } else {
                subtokens = [token]
            }
            subtokens.last?.whitespace = token.whitespace

            // 이 토큰이 대시일 때, 앞뒤가 공백 없이 붙어 있는가(= 단어를 잇는 하이픈인가).
            let dashJoinsWords = token.whitespace.isEmpty
                && i > 0 && tokens[i - 1].whitespace.isEmpty

            for j in 0..<subtokens.count {
                let token = subtokens[j]

                if token.meta.alias != nil || token.phonemes != nil {
                    // Already resolved
                } else if token.tag == .otherWord, Lexicon.currencies[token.text] != nil {
                    currency = token.text
                    token.phonemes = ""
                    token.meta.rating = 4
                } else if token.tag == .dash
                    || (token.tag == .punctuation && token.text == "–")
                {
                    // 단어 내부 하이픈(re-use, well-known)은 붙임표지 쉼표가 아니다. `—`를 주면
                    // Kokoro가 그 자리에서 멈춰 "well [쉼] known"처럼 읽는다. 앞뒤에 공백이 없으면
                    // 단어를 잇는 하이픈으로 보고 음소를 비우고, 독립된 대시("a - b", "a — b")만
                    // 쉼으로 남긴다.
                    token.phonemes = dashJoinsWords ? "" : "—"
                    token.meta.rating = 3
                } else if let tag = token.tag, EnglishG2P.punctuationTags.contains(tag),
                    !token.text.lowercased().unicodeScalars.allSatisfy({
                        (97...122).contains(Int($0.value))
                    })
                {
                    if let val = EnglishG2P.punctuationTagPhonemes[token.text] {
                        token.phonemes = val
                    } else if token.tag == .openQuote {
                        token.phonemes = "\u{201C}"
                    } else if token.tag == .closeQuote {
                        token.phonemes = "\u{201D}"
                    } else {
                        token.phonemes = token.text.filter {
                            EnglishG2P.punctuations.contains($0)
                        }
                    }
                    token.meta.rating = 4
                } else if currency != nil {
                    let looksNumeric = token.text.contains(where: { $0.isNumber })
                    if token.tag != .number && !looksNumeric {
                        currency = nil
                    } else if j + 1 == subtokens.count
                        && (i + 1 == tokens.count || tokens[i + 1].tag != .number)
                    {
                        token.meta.currency = currency
                    }
                } else if j > 0 && j < subtokens.count - 1 && token.text == "2" {
                    let prev = subtokens[j - 1].text
                    let next = subtokens[j + 1].text
                    if (prev.last.map { String($0) } ?? ""
                        + (next.first.map { String($0) } ?? "")).allSatisfy({
                            $0.isLetter
                        })
                        || (prev == "-" && next == "-")
                    {
                        token.meta.alias = "to"
                    }
                }

                // Re-tag otherWord tokens that look numeric (e.g. "98.6", "19.99")
                // so they flow through the number handling path in transcribe()
                if token.tag == .otherWord,
                    token.text.first?.isNumber == true,
                    token.text.allSatisfy({ $0.isNumber || $0 == "." || $0 == "," })
                {
                    token.tag = .number
                }

                if token.meta.alias != nil || token.phonemes != nil {
                    words.append(.single(token))
                } else if case .compound(let last) = words.last,
                    last.last?.whitespace.isEmpty == true
                {
                    var arr = last
                    token.meta.is_head = false
                    arr.append(token)
                    _ = words.popLast()
                    words.append(.compound(arr))
                } else {
                    if token.whitespace.isEmpty { words.append(.compound([token])) } else {
                        words.append(.single(token))
                    }
                }
            }
        }

        return words.map { item in
            if case .compound(let arr) = item, arr.count == 1 {
                return .single(arr[0])
            }
            return item
        }
    }

    // MARK: - CamelCase Fallback (replaces BART neural network)

    /// Split a word on CamelCase / compound boundaries.
    /// "AVFoundation" → ["AV", "Foundation"], "viewDidLoad" → ["view", "Did", "Load"]
    private func splitCamelCase(_ word: String) -> [String] {
        let ns = word as NSString
        let range = NSRange(location: 0, length: ns.length)
        let matches = Self.camelSplitRegex.matches(in: word, options: [], range: range)
        let parts = matches.map { ns.substring(with: $0.range) }
        return parts.isEmpty ? [word] : parts
    }

    /// OOV fallback: CamelCase splitting with per-part resolution.
    ///
    /// Each part is resolved independently: lexicon → BART → letter spelling.
    /// "AVKubernetesPlayer" → "AV"(spell) + "Kubernetes"(BART) + "Player"(lexicon)
    private func fallback(_ word: MToken) -> (phoneme: String?, rating: Int?) {
        let text = word.text

        let parts = splitCamelCase(text)
        if parts.count > 1 {
            let fragments = parts.map { resolvePart($0) }
            let joined = fragments.map(\.0).joined(separator: " ")
            let minRating = fragments.map(\.1).min() ?? 1
            return (joined, minRating)
        }

        return resolvePart(text)
    }

    /// Resolve a single word/part through the full fallback chain:
    /// lexicon → letter spelling (for acronyms) → BART → letter spelling → raw text.
    private func resolvePart(_ text: String) -> (String, Int) {
        if let ph = lexicon.phonemesForWord(text) {
            return (ph, 3)
        }

        // All-uppercase short strings are likely acronyms — spell them out
        if text.count <= 4, text == text.uppercased(), text.allSatisfy({ $0.isLetter }) {
            let nnp = lexicon.getNNP(text)
            if let ph = nnp.0 { return (ph, nnp.1 ?? 2) }
        }

        if let result = bart?.predict(text.lowercased()) {
            return (result, 2)
        }

        let nnp = lexicon.getNNP(text)
        if let phoneme = nnp.phoneme {
            return (phoneme, nnp.rating ?? 2)
        }

        return (unk, 1)
    }

    // MARK: - Main Pipeline

    // swiftlint:disable:next function_body_length
    func phonemize(text: String, performPreprocess: Bool = true) -> (String, [MToken]) {
        let pre: PreprocessTuple
        if performPreprocess {
            pre = self.preprocess(text: text)
        } else {
            pre = (text: text, tokens: [], features: [])
        }

        var tokens = tokenize(preprocessedText: pre)
        tokens = foldLeft(tokens)

        let words = retokenize(tokens)

        var ctx = TokenContext()
        for i in stride(from: words.count - 1, through: 0, by: -1) {
            switch words[i] {
            case .single(let w):
                if w.phonemes == nil {
                    let out = lexicon.transcribe(w, ctx: ctx)
                    w.phonemes = out.0
                    w.meta.rating = out.1
                }

                if w.phonemes == nil {
                    let out = fallback(w)
                    w.phonemes = out.0
                    w.meta.rating = out.1
                }

                ctx = tokenContext(ctx, ps: w.phonemes, token: w)

            case .compound(var arr):
                var left = 0
                var right = arr.count
                var shouldFallback = false
                while left < right {
                    let hasFixed = arr[left..<right].contains {
                        $0.meta.alias != nil || $0.phonemes != nil
                    }
                    let token: MToken? =
                        hasFixed ? nil : mergeTokens(Array(arr[left..<right]))
                    let res: (String?, Int?) =
                        (token == nil) ? (nil, nil) : lexicon.transcribe(token!, ctx: ctx)

                    if let phonemes = res.0 {
                        arr[left].phonemes = phonemes
                        arr[left].meta.rating = res.1
                        for j in (left + 1)..<right {
                            arr[j].phonemes = ""
                            arr[j].meta.rating = res.1
                        }
                        ctx = tokenContext(ctx, ps: phonemes, token: token!)
                        right = left
                        left = 0
                    } else if left + 1 < right {
                        left += 1
                    } else {
                        right -= 1
                        let last = arr[right]
                        if last.phonemes == nil {
                            if last.text.allSatisfy({
                                EnglishG2P.subTokenJunks.contains($0)
                            }) {
                                last.phonemes = ""
                                last.meta.rating = 3
                            } else {
                                shouldFallback = true
                                break
                            }
                        }
                        left = 0
                        arr[right] = last
                    }
                }

                if shouldFallback {
                    let token = mergeTokens(arr)
                    let first = arr[0]
                    let out = fallback(token)
                    first.phonemes = out.0
                    first.meta.rating = out.1
                    arr[0] = first
                    if arr.count > 1 {
                        for j in 1..<arr.count {
                            arr[j].phonemes = ""
                            arr[j].meta.rating = out.1
                        }
                    }
                } else {
                    resolveTokens(&arr)
                }
            }
        }

        let finalTokens: [MToken] = words.map { item in
            switch item {
            case .single(let token):
                return token
            case .compound(let arr):
                return mergeTokens(arr, unk: self.unk)
            }
        }

        for i in 0..<finalTokens.count {
            if var ps = finalTokens[i].phonemes, !ps.isEmpty {
                ps = ps.replacingOccurrences(of: "ɾ", with: "T")
                    .replacingOccurrences(of: "ʔ", with: "t")
                finalTokens[i].phonemes = ps
            }
        }

        let result = finalTokens.map { ($0.phonemes ?? self.unk) + $0.whitespace }.joined()
        return (result, finalTokens)
    }
}
