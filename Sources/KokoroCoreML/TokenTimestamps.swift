import Foundation

/// 단어(G2P 토큰) 단위 시작/끝 타임스탬프.
///
/// Kokoro의 duration predictor가 내는 **음소별 프레임 수**(`pred_dur`)와 G2P가 만든 `[MToken]`을
/// 맞물려 계산한다. 두 입력 모두 이 패키지 안에 이미 존재한다 —
///   · `pred_dur`: CoreML frontend 출력(`pred_dur_clamped`) 또는 7-stage PostAlbert의 `duration`을
///     `max(1, round(·))` 클램프한 값
///   · `[MToken]`: `EnglishG2P.phonemize(text:)`의 두 번째 반환값(지금까지 버려지던 것)
///
/// 알고리즘은 MLX 백엔드(`mlalma/kokoro-ios`의 `TimestampPredictor`, Apache-2.0)를 그대로 옮긴 것이다.
/// 원본은 `MLXArray`를 받지만 텐서 연산은 하나도 쓰지 않아(원소 읽기와 구간 합뿐) `[Int]`로 치환하면
/// **수치가 동일**하다. 상수도 동일하다: 24 kHz ÷ hop 600 = 40 frames/s, 공백을 반으로 쪼개려고
/// 반프레임 단위로 세므로 나눗수는 80.
public struct KokoroTokenTimestamp: Sendable, Equatable {
    /// 원문에 나타난 그대로의 단어 텍스트.
    public let text: String
    /// 이 청크 오디오 기준 시작 시각(초).
    public let start: Double
    /// 이 청크 오디오 기준 끝 시각(초).
    public let end: Double
}

enum TokenTimestampPredictor {
    /// 반프레임 → 초. 24000 / 600 = 40 frames/s, 반프레임이므로 ×2.
    private static let halfFramesPerSecond: Double = 80.0

    /// `tokens`의 `start_ts`/`end_ts`를 **제자리에서** 채운다.
    ///
    /// - Parameters:
    ///   - tokens: `EnglishG2P`가 만든 토큰들. `phonemes`가 nil인 토큰(구두점·공백 등)은 건너뛰되
    ///     그만큼 duration 인덱스를 전진시킨다.
    ///   - predDur: 음소별 프레임 수. 인덱스 0은 BOS, 마지막은 EOS라 최소 3개가 필요하다.
    ///     `tokens`의 음소 총합 + 2와 길이가 맞아야 한다(토크나이저가 IPA 문자 1개 = 토큰 1개).
    static func annotate(tokens: [MToken], predDur: [Int]) {
        guard !tokens.isEmpty, predDur.count >= 3 else { return }

        // 반프레임 단위로 (left, right) 두 커서를 굴린다. 이렇게 해야 단어 사이 공백을
        // 앞 단어의 꼬리와 뒷 단어의 머리에 절반씩 나눠 줄 수 있다.
        // BOS(-3)는 원본 구현의 오프셋을 그대로 유지한다.
        var right = Double(2 * max(0, predDur[0] - 3))
        var left = right

        var i = 1
        for token in tokens {
            guard i < predDur.count - 1 else { break }

            guard let phonemes = token.phonemes else {
                // 음소가 없는 토큰. 공백을 물고 있으면 그 공백 프레임만큼 커서를 민다.
                if !token.whitespace.isEmpty {
                    i += 1
                    guard i < predDur.count else { break }
                    left = right + Double(predDur[i])
                    right = left + Double(predDur[i])
                    i += 1
                }
                continue
            }

            // `Tokenizer.encode`는 vocab에 없는 문자를 버리고 `maxLength`에서 자른다. 그래서 토큰이
            // 들고 있는 음소 수보다 `predDur`가 짧을 수 있다(공백 없는 초장문 단어에서 흔하다).
            // 예전엔 여기서 그냥 break 해 타임스탬프가 **하나도** 안 나왔다 — 소리는 나는데 하이라이트만
            // 죽는 상태다. 남은 프레임까지만이라도 배분하고 끝내는 편이 훨씬 낫다.
            let j = min(i + phonemes.count, predDur.count - 1)
            guard j > i else { break }

            token.start_ts = left / halfFramesPerSecond

            let tokenFrames = predDur[i..<j].reduce(0, +)
            let spaceFrames = token.whitespace.isEmpty ? 0 : predDur[j]

            left = right + Double(2 * tokenFrames + spaceFrames)
            token.end_ts = left / halfFramesPerSecond
            right = left + Double(spaceFrames)

            i = j + (token.whitespace.isEmpty ? 0 : 1)
        }
    }

    /// `annotate`를 돌린 뒤 타임스탬프가 채워진 토큰만 공개 타입으로 추린다.
    ///
    /// - Parameter offset: 이 청크 오디오가 전체 합성물에서 시작하는 시각(초). 청크를 이어붙이는
    ///   호출부가 누적 오프셋을 넘긴다.
    /// - Parameter leadingTrimSeconds: 합성 후 앞에서 잘라낸 오디오 길이(초). duration은 자르기 전
    ///   기준이므로 그만큼 빼야 타임라인이 실제 샘플과 맞는다.
    static func timestamps(
        from tokens: [MToken],
        predDur: [Int],
        offset: Double = 0,
        leadingTrimSeconds: Double = 0
    ) -> [KokoroTokenTimestamp] {
        annotate(tokens: tokens, predDur: predDur)
        return tokens.compactMap { token in
            guard let start = token.start_ts, let end = token.end_ts else { return nil }
            let s = max(0, start - leadingTrimSeconds) + offset
            let e = max(s, end - leadingTrimSeconds + offset)
            return KokoroTokenTimestamp(text: token.text, start: s, end: e)
        }
    }
}
