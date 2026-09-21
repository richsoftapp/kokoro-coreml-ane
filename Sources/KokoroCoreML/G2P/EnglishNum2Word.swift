// Originally from MisakiSwift by mlalma, Apache License 2.0

import Foundation

struct EnglishNum2Word {
    enum ConversionFormat {
        case ordinal
        case ordinalNum
        case decimal
        case year
    }

    private let negWord = "minus "
    private let pointWord = "point"

    private let midNumWords: [(Int, String)] = [
        (1000, "thousand"), (100, "hundred"),
        (90, "ninety"), (80, "eighty"), (70, "seventy"),
        (60, "sixty"), (50, "fifty"), (40, "forty"),
        (30, "thirty"), (20, "twenty"),
    ]

    private let lowNumWords = [
        "twenty", "nineteen", "eighteen", "seventeen",
        "sixteen", "fifteen", "fourteen", "thirteen",
        "twelve", "eleven", "ten", "nine", "eight",
        "seven", "six", "five", "four", "three", "two",
        "one", "zero",
    ]

    private let ords: [String: String] = [
        "one": "first", "two": "second", "three": "third",
        "four": "fourth", "five": "fifth", "six": "sixth",
        "seven": "seventh", "eight": "eighth", "nine": "ninth",
        "ten": "tenth", "eleven": "eleventh", "twelve": "twelfth",
    ]

    private var cards: [Int: String] = [:]

    init() {
        var cards: [Int: String] = [:]
        let highWords = ["m", "b", "tr", "quadr", "quint", "sext", "sept", "oct", "non", "dec"]
        for (index, word) in highWords.enumerated() {
            let power = 6 + (index * 3)
            let val = pow(10.0, Double(power))
            if val <= Double(Int.max) {
                let intVal: Int = Int(val)
                cards[intVal] = word + "illion"
            }
        }
        self.cards = cards
    }

    private func merge(_ lPair: (String, Int), _ rPair: (String, Int)) -> (String, Int) {
        let (lText, lNum) = lPair
        let (rText, rNum) = rPair

        if lNum == 1 && rNum < 100 {
            return (rText, rNum)
        } else if lNum < 100 && lNum > rNum {
            return ("\(lText)-\(rText)", lNum + rNum)
        } else if lNum >= 100 && rNum < 100 {
            return ("\(lText) and \(rText)", lNum + rNum)
        } else if rNum > lNum {
            return ("\(lText) \(rText)", lNum * rNum)
        }
        return ("\(lText), \(rText)", lNum + rNum)
    }

    /// `NSDecimalNumber.intValue`는 Int 범위를 넘는 값에서 **감싸진 쓰레기**를 돌려준다 — 2^63은 정확히
    /// `Int.min`이 되고, 그러면 `toCardinal`의 `abs(Int.min)`이 산술 오버플로로 트랩한다
    /// ("9223372036854775808.5" 한 토큰에 합성 프로세스가 죽었다). Int 범위 안일 때만 값을 돌려준다.
    private static func exactInt(_ decimal: Decimal) -> Int? {
        guard decimal.isFinite, decimal.magnitude <= Decimal(Int.max) else { return nil }
        return NSDecimalNumber(decimal: decimal).intValue
    }

    /// Int로 못 담는 큰 수는 자리 숫자를 하나씩 읽는다("nine two two three …") — 크래시나 엉뚱한
    /// 값("four quintillion …")보다 낫다.
    private func digitsSpelled(_ decimal: Decimal) -> String {
        "\(decimal)".compactMap { ch -> String? in
            if let digit = ch.wholeNumberValue, (0...9).contains(digit) { return lowNumWords[20 - digit] }
            if ch == "." { return pointWord }
            if ch == "-" { return negWord.trimmingCharacters(in: .whitespaces) }
            return nil
        }.joined(separator: " ")
    }

    private func toOrdinal(_ decimalNumber: Decimal) -> String {
        guard let number = Self.exactInt(decimalNumber), number > 0 else { return "" }

        var outWords = toCardinal(number).components(separatedBy: " ")
        var lastWords = outWords[outWords.count - 1].components(separatedBy: "-")
        var lastWord = lastWords[lastWords.count - 1].lowercased()

        if let ordinalWord = ords[lastWord] {
            lastWord = ordinalWord
        } else {
            if lastWord.hasSuffix("y") {
                lastWord = String(lastWord.dropLast()) + "ie"
            }
            lastWord += "th"
        }

        lastWords[lastWords.count - 1] = lastWord
        outWords[outWords.count - 1] = lastWords.joined(separator: "-")
        return outWords.joined(separator: " ")
    }

    private func toOrdinalNum(_ decimalNumber: Decimal) -> String {
        guard let number = Self.exactInt(decimalNumber) else { return "" }
        let ordinal = toOrdinal(decimalNumber)
        if ordinal.count >= 2 {
            let suffix = String(ordinal.suffix(2))
            return "\(number)\(suffix)"
        } else {
            return ""
        }
    }

    private func toCardinal(_ number: Int) -> String {
        if number < 0 {
            // `abs(Int.min)`은 오버플로 트랩 — 부호를 뒤집을 수 없는 유일한 값은 자리로 읽는다.
            guard number != Int.min else { return negWord + digitsSpelled(Decimal(number).magnitude) }
            return negWord + toCardinal(abs(number))
        }

        if number < 21 {
            return lowNumWords[20 - number]
        }

        if number < 100 {
            let tens = (number / 10) * 10
            let ones = number % 10
            if ones == 0 {
                return midNumWords.first { $0.0 == tens }?.1 ?? ""
            } else {
                let tensWord = midNumWords.first { $0.0 == tens }?.1 ?? ""
                let onesWord = lowNumWords[20 - ones]
                return "\(tensWord)-\(onesWord)"
            }
        }

        if number < 1000 {
            let hundreds = number / 100
            let remainder = number % 100
            let hundredsWord = toCardinal(hundreds) + " hundred"
            if remainder == 0 {
                return hundredsWord
            } else {
                return "\(hundredsWord) and \(toCardinal(remainder))"
            }
        }

        // 큰 단위(million 이상)를 **먼저** 본다. midNumWords에는 1000(thousand)이 들어 있어서
        // 이 순서가 뒤집히면 1,000,000이 1000에 먼저 걸려 quotient가 1000이 되고,
        // toCardinal(1000)="one thousand"가 붙어 "one thousand thousand"가 된다(백만이 사라짐).
        let scales = cards.map { ($0.key, $0.value) }.sorted { $0.0 > $1.0 }
            + midNumWords.sorted { $0.0 > $1.0 }
        for (value, word) in scales {
            if number >= value {
                let quotient = number / value
                let remainder = number % value
                let quotientWord = toCardinal(quotient)
                if remainder == 0 {
                    return "\(quotientWord) \(word)"
                } else {
                    return "\(quotientWord) \(word), \(toCardinal(remainder))"
                }
            }
        }

        return ""
    }

    private func toYear(_ yearDecimal: Decimal, suffix: String? = nil, longVal: Bool = true) -> String {
        guard let year = Self.exactInt(yearDecimal), year != Int.min else { return digitsSpelled(yearDecimal) }
        var val = year
        var finalSuffix = suffix

        if val < 0 {
            val = abs(val)
            finalSuffix = finalSuffix ?? "BC"
        }

        let high = val / 100
        let low = val % 100

        let valText: String
        if high == 0 || (high % 10 == 0 && low < 10) || high >= 100 {
            valText = toCardinal(val)
        } else {
            let highText = toCardinal(high)
            let lowText: String
            if low == 0 {
                lowText = "hundred"
            } else if low < 10 {
                lowText = "oh-\(toCardinal(low))"
            } else {
                lowText = toCardinal(low)
            }
            valText = "\(highText) \(lowText)"
        }

        if let suffix = finalSuffix {
            return "\(valText) \(suffix)"
        } else {
            return valText
        }
    }

    private func toDecimal(_ number: Decimal) -> String {
        guard let integerPart = Self.exactInt(number) else { return digitsSpelled(number) }
        let fractionalPart = number - Decimal(integerPart)

        if fractionalPart == 0 {
            return toCardinal(integerPart)
        }

        let integerWords = toCardinal(integerPart)

        let fractionalString = "\(fractionalPart)".dropFirst(2)
        let fractionalWords = fractionalString.map { toCardinal(Int(String($0)) ?? 0) }.joined(
            separator: " ")

        return "\(integerWords) \(pointWord) \(fractionalWords)"
    }

    func convert(_ number: Decimal, to format: ConversionFormat = .decimal) -> String {
        switch format {
        case .ordinal:
            return toOrdinal(number)
        case .ordinalNum:
            return toOrdinalNum(number)
        case .year:
            return toYear(number)
        case .decimal:
            return toDecimal(number)
        }
    }
}
