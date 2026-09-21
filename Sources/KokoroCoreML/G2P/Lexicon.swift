// Originally from MisakiSwift by mlalma, Apache License 2.0

import Foundation
import NaturalLanguage

final class Lexicon {  // swiftlint:disable:this type_body_length
    static let usVocab: Set<Character> = Set("AIOWYbdfhijklmnpstuvwzæðŋɑɔəɛɜɡɪɹɾʃʊʌʒʤʧˈˌθᵊᵻʔ")
    static let gbVocab: Set<Character> = Set("AIQWYabdfhijklmnpstuvwzðŋɑɒɔəɛɜɡɪɹʃʊʌʒʤʧˈˌːθᵊ")
    static let lexiconOrdinals: [Int] = [39, 45] + Array(65...90) + Array(97...122)
    static let ordinals: Set<String> = Set(["st", "nd", "rd", "th"])

    static let addSymbols: [String: String] = [".": "dot", "/": "slash"]
    static let primaryStress: Character = "ˈ"
    static let secondaryStress: Character = "ˌ"
    static let vowelSet: Set<Character> = Set("AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ")
    static let symbolSet: [String: String] = ["%": "percent", "&": "and", "+": "plus", "@": "at"]
    static let usTaus: Set<Character> = Set("AIOWYiuæɑəɛɪɹʊʌ")
    static let currencies: [String: (String, String)] = [
        "$": ("dollar", "cent"),
        "£": ("pound", "pence"),
        "€": ("euro", "cent"),
        "¥": ("yen", "sen"),
        "₩": ("won", "jeon"),
    ]

    private let british: Bool
    private let capStresses: (Double, Double) = (0.5, 2.0)
    private let num2Words = EnglishNum2Word()

    private let golds: [String: Any]
    private let silvers: [String: Any]
    private let vocab: Set<Character>

    init(british: Bool) {
        self.british = british
        let rawGolds = DataResourcesUtil.loadGold(british: british)
        let rawSilvers = DataResourcesUtil.loadSilver(british: british)
        self.golds = Lexicon.growDictionary(rawGolds)
        self.silvers = Lexicon.growDictionary(rawSilvers)

        self.vocab = british ? Lexicon.gbVocab : Lexicon.usVocab
    }

    private static func growDictionary(_ d: [String: Any]) -> [String: Any] {
        var e: [String: Any] = [:]

        for (k, v) in d {
            if k.count < 2 {
                continue
            }

            if k == k.lowercased() {
                if k != k.capitalized {
                    e[k.capitalized] = v
                }
            } else if k == k.lowercased().capitalized {
                e[k.lowercased()] = v
            }
        }

        return e.merging(d) { _, original in original }
    }

    static func applyStress(_ phoneticString: String?, stress: Double?) -> String? {
        func restress(_ ps: String) -> String {
            let characters = Array(ps)
            var indexedChars: [(Double, Character)] = characters.enumerated().map {
                (Double($0), $1)
            }

            var stressToVowel: [Int: Int] = [:]
            for (i, char) in characters.enumerated() {
                if stresses.contains(char) {
                    for j in (i + 1)..<characters.count {
                        if Lexicon.vowelSet.contains(characters[j]) {
                            stressToVowel[i] = j
                            break
                        }
                    }
                }
            }

            for (stressIndex, vowelIndex) in stressToVowel {
                let stressChar = indexedChars[stressIndex].1
                indexedChars[stressIndex] = (Double(vowelIndex) - 0.5, stressChar)
            }

            return String(indexedChars.sorted { $0.0 < $1.0 }.map { $0.1 })
        }

        guard let phoneticString else { return nil }
        guard let stress else { return phoneticString }

        let stresses = Set<Character>([Lexicon.primaryStress, Lexicon.secondaryStress])

        if stress < -1 {
            return
                phoneticString.replacingOccurrences(of: String(Lexicon.primaryStress), with: "")
                .replacingOccurrences(of: String(Lexicon.secondaryStress), with: "")
        } else if stress == -1
            || (stress == 0 || stress == -0.5) && phoneticString.contains(Lexicon.primaryStress)
        {
            return
                phoneticString.replacingOccurrences(of: String(Lexicon.secondaryStress), with: "")
                .replacingOccurrences(
                    of: String(Lexicon.primaryStress), with: String(Lexicon.secondaryStress))
        } else if (stress == 0 || stress == 0.5 || stress == 1)
            && !phoneticString.contains(where: { stresses.contains($0) })
        {
            if !phoneticString.contains(where: { Lexicon.vowelSet.contains($0) }) {
                return phoneticString
            }
            return restress(String(Lexicon.secondaryStress) + phoneticString)
        } else if stress >= 1 && !phoneticString.contains(Lexicon.primaryStress)
            && phoneticString.contains(Lexicon.secondaryStress)
        {
            return phoneticString.replacingOccurrences(
                of: String(Lexicon.secondaryStress), with: String(Lexicon.primaryStress))
        } else if stress > 1 && !phoneticString.contains(where: { stresses.contains($0) }) {
            if !phoneticString.contains(where: { Lexicon.vowelSet.contains($0) }) {
                return phoneticString
            }
            return restress(String(Lexicon.primaryStress) + phoneticString)
        }

        return phoneticString
    }

    func transcribe(_ token: MToken, ctx: TokenContext) -> (String?, Int?) {
        var word = token.text
        if let alias = token.meta.alias { word = alias }
        word =
            word.replacingOccurrences(of: String(UnicodeScalar(8216)!), with: "'")
            .replacingOccurrences(of: String(UnicodeScalar(8217)!), with: "'")
        word = word.precomposedStringWithCompatibilityMapping

        word = String(word.map { unicodeNumericIfNeeded($0) })

        let stress: Double? =
            (word == word.lowercased()
                ? nil : (word == word.uppercased() ? capStresses.1 : capStresses.0))
        let res = getWord(word, tag: token.pos, stress: stress, ctx: ctx)
        if let phoneme = res.phoneme {
            return (
                Lexicon.applyStress(
                    appendCurrency(phoneme, currency: token.meta.currency), stress: token.meta.stress),
                res.rating
            )
        } else if isNumber(word: word, is_head: token.meta.is_head) {
            let num = getNumber(
                word, currency: token.meta.currency, is_head: token.meta.is_head,
                num_flags: token.meta.num_flags)
            return (Lexicon.applyStress(num.0, stress: token.meta.stress), num.1)
        } else if !word.unicodeScalars.allSatisfy({
            Lexicon.lexiconOrdinals.contains(Int($0.value))
        }) {
            return (nil, nil)
        }

        return (nil, nil)
    }

    // MARK: - Internal (for CamelCase fallback)

    /// Look up a single word's phonemes from the gold/silver dictionaries.
    /// 품사별 항목(reuse: DEFAULT/NOUN)은 `tag`가 있으면 그 태그 → 상위 품사 → DEFAULT 순으로 고른다
    /// ([[lookup(_:tag:stress:ctx:)]]와 같은 규칙). 없으면 DEFAULT.
    func phonemesForWord(_ word: String, tag: String? = nil) -> String? {
        let lc = word.lowercased()
        func pick(_ dict: [String: String?]) -> String? {
            if let tag, let v = dict[tag] as? String { return v }
            if let parent = Lexicon.parentTag(tag), let v = dict[parent] as? String { return v }
            return dict["DEFAULT"] as? String
        }
        if let v = golds[lc] as? String { return v }
        if let v = golds[word] as? String { return v }
        if let dict = golds[lc] as? [String: String?], let v = pick(dict) { return v }
        if let v = silvers[lc] as? String { return v }
        if let v = silvers[word] as? String { return v }
        if let dict = silvers[lc] as? [String: String?], let v = pick(dict) { return v }
        return nil
    }

    /// Spell out acronyms and proper nouns letter-by-letter.
    /// 알려진 최상위 도메인. 이 목록에 걸릴 때만 도메인으로 취급해 오탐을 막는다.
    static let knownTLDs: Set<String> = [
        "com", "org", "net", "edu", "gov", "mil", "int", "io", "co", "ai", "app", "dev",
        "me", "info", "biz", "tv", "xyz", "kr", "jp", "uk", "us", "de", "fr", "cn", "ru",
        "ca", "au", "in", "it", "es", "nl", "se", "no", "br", "mx",
    ]

    func getNNP(_ word: String) -> (phoneme: String?, rating: Int?) {
        let pieces: [String?] = word.compactMap { ch in
            if ch.isLetter {
                let s = String(ch).uppercased()
                if let v = golds[s] as? String { return v }
            }
            return nil
        }

        if pieces.contains(where: { $0 == nil }) { return (nil, nil) }

        let joined = Lexicon.applyStress(
            pieces.compactMap { $0 }.joined(separator: ""), stress: 0)
        if let joined {
            let ps = joined.replacingLastOccurrence(
                of: Lexicon.secondaryStress, with: Lexicon.primaryStress)
            return (ps, 3)
        }

        return (nil, nil)
    }

    // MARK: - Private

    private func unicodeNumericIfNeeded(_ c: Character) -> Character {
        guard c.isNumber else { return c }

        if let numericValue = c.wholeNumberValue {
            if numericValue >= 0 && numericValue <= 9 {
                return Character("\(numericValue)")
            }
        }

        return c
    }

    private func getWord(
        _ word: String, tag: String?, stress: Double?, ctx: TokenContext
    )
        -> (phoneme: String?, rating: Int?)
    {
        let sc = getSpecialCase(word, tag: tag, stress: stress, ctx: ctx)
        if sc.phoneme != nil { return sc }
        var candidate = word
        let wl = word.lowercased()

        if word.count > 1,
            word.replacingOccurrences(of: "'", with: "").allSatisfy({ $0.isLetter }),
            word != word.lowercased(),
            (tag != "NNP" || word.count > 7),
            golds[word] == nil, silvers[word] == nil,
            (word == word.uppercased() || word.dropFirst().lowercased() == word.dropFirst()),
            (golds[wl] != nil || silvers[wl] != nil
                || [stem_s, stem_ed, stem_ing].contains(where: { fn in
                    fn(wl, tag, stress, ctx).0 != nil
                }))
        {
            candidate = wl
        }

        if isKnown(candidate) {
            return lookup(candidate, tag: tag, stress: stress, ctx: ctx)
        } else if candidate.hasSuffix("s'"),
            isKnown(String(candidate.dropLast(2)) + "'s")
        {
            return lookup(
                String(candidate.dropLast(2)) + "'s", tag: tag, stress: stress, ctx: ctx)
        } else if candidate.hasSuffix("'"), isKnown(String(candidate.dropLast())) {
            return lookup(String(candidate.dropLast()), tag: tag, stress: stress, ctx: ctx)
        }

        let s = stem_s(candidate, tag: tag, stress: stress, ctx: ctx)
        if s.phoneme != nil { return s }

        let ed = stem_ed(candidate, tag: tag, stress: stress, ctx: ctx)
        if ed.phoneme != nil { return ed }

        let ing = stem_ing(candidate, tag: tag, stress: (stress == nil ? 0.5 : stress), ctx: ctx)
        if ing.phoneme != nil { return ing }

        return (nil, nil)
    }

    // swiftlint:disable:next cyclomatic_complexity
    private func getSpecialCase(
        _ word: String, tag: String?, stress: Double?, ctx: TokenContext
    )
        -> (phoneme: String?, rating: Int?)
    {
        if let tag, Lexicon.punctuationTags.contains(tag), let target = Lexicon.addSymbols[word] {
            return lookup(target, tag: nil, stress: -0.5, ctx: ctx)
        } else if let sym = Lexicon.symbolSet[word] {
            return lookup(sym, tag: nil, stress: nil, ctx: ctx)
        } else if word.trimmingCharacters(in: CharacterSet(charactersIn: ".")).contains("."),
            !word.contains(where: { $0.isNumber })
        {
            let parts = word.split(separator: ".")
            if parts.map({ $0.count }).max() ?? 0 < 3 {
                return getNNP(word)
            }
        } else if word == "a" || word == "A" {
            // 관사(DT)면 ɐ, 아니면 글자 이름 ˈA("Exhibit A"). 품사가 없으면 관사로 — 산문에서 홀로
            // 선 "a"는 거의 언제나 관사고, 글자 이름 "에이"로 읽는 쪽이 훨씬 큰 사고다.
            if tag == "DT" || tag == nil { return ("ɐ", 4) }
            return ("ˈA", 4)
        } else if ["am", "Am", "AM"].contains(word) {
            if let tag, tag.hasPrefix("NN") {
                return getNNP(word)
            }

            if ctx.futureVowel == nil || word != "am" || (stress != nil && stress! > 0) {
                if let v = golds["am"] as? String { return (v, 4) }
            }
            return ("ɐm", 4)
        } else if ["an", "An", "AN"].contains(word) {
            if word == "AN", let tag, tag.hasPrefix("NN") {
                return getNNP(word)
            }
            return ("ɐn", 4)
        } else if word == "I", tag == "PRP" {
            return (String(Lexicon.secondaryStress) + "I", 4)
        } else if ["by", "By", "BY"].contains(word),
            Lexicon.parentTag(tag) == "ADV"
        {
            return ("bˈI", 4)
        } else if ["to", "To"].contains(word) || (word == "TO" && (tag == "TO" || tag == "IN")) {
            let chosen: String
            if ctx.futureVowel == nil {
                chosen = (golds["to"] as? String) ?? "to"
            } else if ctx.futureVowel == false {
                chosen = "tə"
            } else {
                chosen = "tʊ"
            }
            return (chosen, 4)
        } else if ["in", "In"].contains(word)
            || (word == "IN" && tag != "NNP")
        {
            let s =
                (ctx.futureVowel == nil || tag != "IN")
                ? String(Lexicon.primaryStress) : ""
            return (s + "ɪn", 4)
        } else if ["the", "The"].contains(word) || (word == "THE" && tag == "DT") {
            return (ctx.futureVowel == true ? "ði" : "ðə", 4)
        } else if tag == "IN",
            word.range(of: "(?i)vs\\.?$", options: .regularExpression) != nil
        {
            return lookup("versus", tag: nil, stress: nil, ctx: ctx)
        } else if ["used", "Used", "USED"].contains(word) {
            if (tag == "VBD" || tag == "JJ") && ctx.futureTo {
                if let m = golds["used"] as? [String: String?], let v = m["VBD"] as? String {
                    return (v, 4)
                }
            }
            if let m = golds["used"] as? [String: String?], let v = m["DEFAULT"] as? String {
                return (v, 4)
            }
        }

        return (nil, nil)
    }

    private func lookup(
        _ w: String, tag: String?, stress: Double?, ctx: TokenContext?
    ) -> (
        phoneme: String?, rating: Int?
    ) {
        var word = w
        var isNNP: Bool? = nil
        if word == word.uppercased(), golds[word] == nil {
            word = word.lowercased()
            isNNP = tag == "NNP"
        }
        var phoneticString: Any? = golds[word]
        var rating = 4
        // 대문자 고유명사(NNP)는 원래 silver를 건너뛰고 철자로 읽었다("IBM" 같은 두문자 방어). 그런데 표지·
        // 머리말의 이름("HOWARD MARKS", "MORGAN HOUSEL")도 전부 대문자 NNP라, silver에만 있는 MARKS·HOUSEL이
        // "M A R K S"로 읽혔다. 4글자 이상이면 silver의 낱말 발음을 쓴다 — 4글자 이상 두문자 중 silver에 낱말로
        // 있는 건(scuba, laser, radar…) 어차피 낱말로 읽는 것들이다.
        let allowsSilverForProperNoun = isNNP == true && w.count >= 4
        if phoneticString == nil, isNNP != true || allowsSilverForProperNoun {
            phoneticString = silvers[word]
            rating = 3
        }
        // 사전에 없는 5글자 이상 대문자 NNP("HOWARD")는 철자가 아니라 BART 폴백(낱말 발음)으로 보낸다 —
        // 5글자 이상 두문자는 대개 낱말처럼 읽고(NASDAQ, UNESCO), 책에선 대문자 이름이 훨씬 흔하다.
        // 4글자 이하는 기존대로 철자(IBM, HDMI). nil을 돌려주면 getWord → 파이프라인 fallback(BART)으로 간다.
        // 단 모음(AEIOUY)이 하나도 없으면(MSNBC, LGBTQ, HTTPS, PBKDF) 낱말로 읽을 수 없는 두문자라 철자로 둔다 —
        // 이름은 모음 없이 5글자 이상일 수 없다(Y만 있는 이름을 위해 Y도 모음으로 친다 — 단 그런 이름은 품사
        // 태거가 고유명사로 보지 않으면 원래 경로대로 철자로 읽힌다).
        if phoneticString == nil, isNNP == true, w.count >= 5, w.allSatisfy(\.isLetter), Lexicon.containsVowel(w) {
            return (nil, nil)
        }

        // Python Misaki와 같은 순서: 뒤에 모음 정보가 없고 "None" 항목이 있으면 그것, 아니면 정확한
        // Penn 태그(DT·VBD·VBP…), 그것도 없으면 상위 품사(VERB·NOUN·ADV·ADJ), 마지막으로 DEFAULT.
        if let phonemeDict = phoneticString as? [String: String?] {
            var t = tag
            if let ctx = ctx, ctx.futureVowel == nil, phonemeDict["None"] != nil {
                t = "None"
            } else if let current = t, phonemeDict[current] == nil {
                t = Lexicon.parentTag(current)
            }
            phoneticString = phonemeDict[t ?? "DEFAULT"] ?? phonemeDict["DEFAULT"]
        }

        if phoneticString == nil
            || (isNNP == true
                && !(phoneticString as? String ?? "").contains(Lexicon.primaryStress))
        {
            let nn = getNNP(word)
            if nn.phoneme != nil { return nn }
        }

        let applied = Lexicon.applyStress(phoneticString as? String, stress: stress)
        return (applied, rating)
    }

    /// 낱말로 발음할 수 있는지의 최소 조건 — 모음 글자(Y 포함)가 하나라도 있는지.
    private static func containsVowel(_ word: String) -> Bool {
        word.lowercased().contains { "aeiouy".contains($0) }
    }

    /// Penn 태그의 상위 품사(Python Misaki `get_parent_tag`). 사전 항목이 VERB/NOUN/ADV/ADJ 키로
    /// 갈라진 낱말(read·lead·live·record…)이 쓴다.
    static func parentTag(_ tag: String?) -> String? {
        guard let tag else { return nil }
        if tag.hasPrefix("VB") { return "VERB" }
        if tag.hasPrefix("NN") { return "NOUN" }
        if tag.hasPrefix("ADV") || tag.hasPrefix("RB") { return "ADV" }
        if tag.hasPrefix("ADJ") || tag.hasPrefix("JJ") { return "ADJ" }
        return tag
    }

    /// Penn 구두점 태그(spaCy의 ADD·PUNCT 자리).
    static let punctuationTags: Set<String> = [".", ",", ":", "(", ")", "``", "''", "#", "$", "SYM", "HYPH", "NFP"]

    private func isKnown(_ word: String) -> Bool {
        // 빈 문자열은 아래 `index(after: startIndex)`가 트랩한다(빈 alias 등으로 들어올 수 있는 값).
        guard !word.isEmpty else { return false }
        if golds[word] != nil || Lexicon.symbolSet[word] != nil || silvers[word] != nil {
            return true
        }

        if !word.allSatisfy({ ch in
            if let v = ch.unicodeScalars.first?.value {
                return Lexicon.lexiconOrdinals.contains(Int(v))
            }
            return false
        }) {
            return false
        }

        if word.count == 1 { return true }
        if word == word.uppercased(), golds[word.lowercased()] != nil { return true }
        let idx = word.index(after: word.startIndex)
        return word[idx...].uppercased() == word[idx...]
    }

    private func stem_s(
        _ word: String, tag: String?, stress: Double?, ctx: TokenContext?
    ) -> (
        phoneme: String?, rating: Int?
    ) {
        guard word.count >= 3, word.hasSuffix("s") else { return (nil, nil) }
        var stem: String?

        if !word.hasSuffix("ss"), isKnown(String(word.dropLast())) {
            stem = String(word.dropLast())
        } else if word.hasSuffix("'s")
            || (word.count > 4 && word.hasSuffix("es") && !word.hasSuffix("ies")),
            isKnown(String(word.dropLast(2)))
        {
            stem = String(word.dropLast(2))
        } else if word.count > 4 && word.hasSuffix("ies"),
            isKnown(String(word.dropLast(3)) + "y")
        {
            stem = String(word.dropLast(3)) + "y"
        }

        guard let s = stem else { return (nil, nil) }
        let looked = lookup(s, tag: tag, stress: stress, ctx: ctx)
        return (pluralizeS(looked.0), looked.1)
    }

    /// [[EnglishG2P.fallback(_:)]]도 쓴다 — 사전에 없는 낱말(고유명사 등) 뒤 소유격 's를
    /// 글자 이름("에스")으로 철자내지 않고 어간의 마지막 소리에 맞는 /s/·/z/·/ɪz/로 잇기 위해.
    func pluralizeS(_ stem: String?) -> String? {
        guard let stem = stem, !stem.isEmpty else { return nil }
        if let last = stem.last, "ptkfθ".contains(last) { return stem + "s" }
        if let last = stem.last, "szʃʒʧʤ".contains(last) {
            return stem + (british ? "ɪ" : "ᵻ") + "z"
        }
        return stem + "z"
    }

    private func pastEd(_ stem: String?) -> String? {
        guard let stem = stem, !stem.isEmpty else { return nil }
        if let last = stem.last, "pkfθʃsʧ".contains(last) { return stem + "t" }
        if stem.hasSuffix("d") { return stem + (british ? "ɪ" : "ᵻ") + "d" }
        if !stem.hasSuffix("t") { return stem + "d" }
        if british || stem.count < 2 { return stem + "ɪd" }
        if let penult = stem.dropLast().last, Lexicon.usTaus.contains(penult) {
            return String(stem.dropLast()) + "ɾᵻd"
        }
        return stem + "ᵻd"
    }

    private func stem_ed(
        _ word: String, tag: String?, stress: Double?, ctx: TokenContext?
    ) -> (
        phoneme: String?, rating: Int?
    ) {
        guard word.count >= 4, word.hasSuffix("d") else { return (nil, nil) }
        var stem: String?

        if !word.hasSuffix("dd"), isKnown(String(word.dropLast())) {
            stem = String(word.dropLast())
        } else if word.count > 4 && word.hasSuffix("ed") && !word.hasSuffix("eed"),
            isKnown(String(word.dropLast(2)))
        {
            stem = String(word.dropLast(2))
        }

        guard let s = stem else { return (nil, nil) }
        let looked = lookup(s, tag: tag, stress: stress, ctx: ctx)
        return (pastEd(looked.0), looked.1)
    }

    private func progIng(_ stem: String?) -> String? {
        guard let stem = stem, !stem.isEmpty else { return nil }

        if british {
            if let last = stem.last, "əː".contains(last) { return nil }
        } else {
            if stem.count > 1, stem.hasSuffix("t"), let penult = stem.dropLast().last,
                Lexicon.usTaus.contains(penult)
            {
                return String(stem.dropLast()) + "ɾɪŋ"
            }
        }

        return stem + "ɪŋ"
    }

    private func stem_ing(
        _ word: String, tag: String?, stress: Double?, ctx: TokenContext?
    ) -> (
        phoneme: String?, rating: Int?
    ) {
        guard word.count >= 5, word.hasSuffix("ing") else { return (nil, nil) }
        var stem: String?

        if word.count > 5, isKnown(String(word.dropLast(3))) {
            stem = String(word.dropLast(3))
        } else if isKnown(String(word.dropLast(3)) + "e") {
            stem = String(word.dropLast(3)) + "e"
        } else if word.count > 5,
            word.range(
                of: #"([bcdgklmnprstvxz])\1ing$|cking$"#, options: .regularExpression)
                != nil, isKnown(String(word.dropLast(4)))
        {
            stem = String(word.dropLast(4))
        }

        guard let s = stem else { return (nil, nil) }
        let looked = lookup(s, tag: tag, stress: stress, ctx: ctx)
        return (progIng(looked.phoneme), looked.rating)
    }

    private func isCurrency(_ word: String) -> Bool {
        if !word.contains(".") { return true }
        if word.filter({ $0 == "." }).count > 1 { return false }
        if let cents = word.split(separator: ".").last {
            return cents.count < 3 || Set(cents) == Set(["0"])
        }

        return false
    }

    private func appendCurrency(_ phoneme: String?, currency: String?) -> String? {
        guard let phoneme, let currency else { return phoneme }

        if let pair = Lexicon.currencies[currency] {
            if let plural = stem_s(pair.0 + "s", tag: nil, stress: nil, ctx: nil).phoneme {
                return phoneme + " " + plural
            }
        }

        return phoneme
    }

    private func isNumber(word: String, is_head: Bool) -> Bool {
        if word.allSatisfy({ !$0.isNumber }) { return false }
        let suffixes: [String] =
            ["ing", "'d", "ed", "'s"] + Lexicon.ordinals + ["s"]
        var core = word
        for s in suffixes {
            if core.hasSuffix(s) {
                core = String(core.dropLast(s.count))
                break
            }
        }

        return core.enumerated().allSatisfy { (i, c) in
            return c.isNumber || c == "," || c == "." || (is_head && i == 0 && c == "-")
        }
    }

    private func isPlainDigits(_ string: String) -> Bool {
        return !string.isEmpty && string.allSatisfy { $0.isNumber }
    }

    // swiftlint:disable:next cyclomatic_complexity function_body_length
    // swiftlint:disable:next cyclomatic_complexity function_body_length
    private func getNumber(
        _ input: String, currency: String?, is_head: Bool, num_flags: String
    ) -> (String?, Int?) {
        var result: [(String, Int)] = []

        func appendLookup(_ w: String, s: Double?) {
            let looked = lookup(w, tag: nil, stress: s, ctx: nil)
            if let p = looked.0, let r = looked.1 { result.append((p, r)) }
        }

        func extend_num(_ num: String, first: Bool = true, escape: Bool = false) {
            let words: String
            if escape {
                words = num
            } else if let val = Decimal(string: num) {
                words = num2Words.convert(val)
            } else {
                words = num
            }
            let splits = words.split(whereSeparator: { !$0.isLetter }).map(String.init)

            for (i, w) in splits.enumerated() {
                if w != "and" || num_flags.contains("&") {
                    if first && i == 0 && splits.count > 1 && w == "one"
                        && num_flags.contains("a")
                    {
                        result.append(("ə", 4))
                    } else {
                        let s = (w == "point") ? -2.0 : nil
                        appendLookup(w, s: s)
                    }
                } else if w == "and" && num_flags.contains("n") && !result.isEmpty {
                    let last = result.removeLast()
                    result.append((last.0 + "ən", last.1))
                }
            }
        }

        var word = input
        var suffix: String? = nil
        if let m = word.range(of: "[a-z']+$", options: .regularExpression) {
            suffix = String(word[m])
            word.removeSubrange(m)
        }

        if word.hasPrefix("-") {
            appendLookup("minus", s: nil)
            word.removeFirst()
        }

        if isPlainDigits(word), let sf = suffix, Lexicon.ordinals.contains(sf) {
            if let n = Int(word) {
                extend_num(num2Words.convert(Decimal(n), to: .ordinal), escape: true)
            }
        } else if result.isEmpty, word.count == 4,
            !Lexicon.currencies.contains(where: { currency == $0.key }), isPlainDigits(word)
        {
            if let n = Int(word) {
                extend_num(num2Words.convert(Decimal(n), to: .year), escape: true)
            }
        } else if !is_head && !word.contains(".") {
            let num = word.replacingOccurrences(of: ",", with: "")
            if num.first == "0" || num.count > 3 {
                for n in num { extend_num(String(n), first: false) }
            } else if num.count == 3 && !num.hasSuffix("00") {
                extend_num(String(num.first!))
                if num[num.index(num.startIndex, offsetBy: 1)] == "0" {
                    if let oh = lookup("O", tag: nil, stress: -2, ctx: nil) as? (String, Int) {
                        result.append(oh)
                    }
                    extend_num(String(num.last!), first: false)
                } else {
                    extend_num(String(num.suffix(2)), first: false)
                }
            } else {
                extend_num(num)
            }
        } else if let curr = currency, let units = Lexicon.currencies[curr], isCurrency(word) {
            var pairs: [(Int, String)] = []
            let parts = word.replacingOccurrences(of: ",", with: "").split(separator: ".")
            let a = parts.indices.contains(0) ? Int(parts[0]) ?? 0 : 0
            let b = parts.indices.contains(1) ? Int(parts[1]) ?? 0 : 0
            pairs = [(a, units.0), (b, units.1)]
            if pairs.count > 1 {
                if pairs[1].0 == 0 {
                    pairs = Array(pairs.prefix(1))
                } else if pairs[0].0 == 0 {
                    pairs = Array(pairs.suffix(1))
                }
            }

            for (i, (num, unit)) in pairs.enumerated() {
                if i > 0 { appendLookup("and", s: nil) }
                extend_num(String(num), first: i == 0)
                if abs(num) != 1 && unit != "pence" {
                    if let s = stem_s(unit + "s", tag: nil, stress: nil, ctx: nil).0 {
                        result.append((s, 4))
                    }
                } else {
                    appendLookup(unit, s: nil)
                }
            }
        } else if word.contains(".") {
            let parts = word.replacingOccurrences(of: ",", with: "").split(separator: ".").map(
                String.init)
            for (idx, num) in parts.enumerated() {
                if idx > 0 { appendLookup("point", s: -2.0) }
                if num.isEmpty {
                } else if idx > 0 || num.first == "0" {
                    for n in num {
                        extend_num(String(n), first: false)
                    }
                } else {
                    extend_num(num, first: idx == 0)
                }
            }
        } else {
            if isPlainDigits(word) {
                if let n = Int(word) { word = num2Words.convert(Decimal(n), to: .decimal) }
            } else if !word.contains(".") {
                let num = word.replacingOccurrences(of: ",", with: "")
                if let n = Int(num) {
                    word = num2Words.convert(
                        Decimal(n),
                        to: (suffix != nil && Lexicon.ordinals.contains(suffix!))
                            ? .ordinal : .decimal)
                }
            } else {
                let num = word.replacingOccurrences(of: ",", with: "")
                if num.first == "." {
                    let tail =
                        num.dropFirst().compactMap { Int(String($0)) }.map {
                            num2Words.convert(Decimal($0))
                        }.joined(separator: " ")
                    word = "point " + tail
                } else {
                    if let d = Double(num) { word = num2Words.convert(Decimal(d)) }
                }
            }

            extend_num(word, escape: true)
        }

        if result.isEmpty { return (nil, nil) }

        var text = result.map { $0.0 }.joined(separator: " ")
        let rating = result.map { $0.1 }.min() ?? 4

        if let s = suffix, s == "s" || s == "'s" {
            text = pluralizeS(text) ?? text
        } else if let s = suffix, s == "ed" || s == "'d" {
            text = pastEd(text) ?? text
        } else if suffix == "ing" {
            text = progIng(text) ?? text
        }

        return (text, rating)
    }
}
