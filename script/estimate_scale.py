import numpy as np
import matplotlib.pyplot as plt
import librosa
from pydub import AudioSegment
import japanize_matplotlib


def estimate_scale(hz):
    """hzから音階を判定する。

    Args:
        hz (_type_): 推定対象のHz
    Returns:
        _type_: 推定音階
    """
    # 音階とHZの配列
    SCALE = ["ラ", "ラ#", "シ", "ド", "ド#", "レ", "レ#", "ミ", "ファ", "ファ#", "ソ", "ソ#"]
    # 基準音440hzのラからの平均律を作成
    hz_array = np.array([440 * (2 ** (i / 12)) for i in range(-24, 24)])
    scale_array = np.array(SCALE * 4)

    # 音がないところはrestにする。
    if hz == 0:
        scale_est = "rest"
    else:
        # Hzの比率が1に近い音階を選択する。
        scale_index = np.argmin(np.abs(np.abs(1 - hz_array / hz)))
        scale_est = scale_array[scale_index]
    return scale_est


def wave2hz(wave, frame_rate):
    """波形から0.1秒ごとにHzを推定する。

    Args:
        wave (_type_): 波形データ
        frame_rate (_type_): 音源のフレームレート

    Returns:
        _type_: 推定Hz
    """
    # 小さな振幅の音は除く
    norm_wave = np.where(np.abs(wave) < 5000, 0, wave)

    # 1秒ずつ、0.1秒ずらしながらFFTを行う。
    fft_size = frame_rate
    fft_wave = np.abs(
        librosa.stft(
            norm_wave.astype(float), n_fft=fft_size, hop_length=frame_rate // 10
        )
    )
    freq = librosa.fft_frequencies(sr=frame_rate, n_fft=fft_size)
    # 1760Hz以上の音階は判定しない。
    hz_list = np.array(
        [freq[max_idx] for max_idx in np.argmax(fft_wave[:1760, :], axis=0)]
    )
    return hz_list


def wav2scale(filepath):
    """wavファイルを音階スケールに変換する

    Args:
        filepath (_type_): _description_

    Returns:
        _type_: _description_
    """

    sound = AudioSegment.from_file(filepath, "wav")
    frame_rate = sound.frame_rate

    # NumPy配列に返還
    wave = np.array(sound.get_array_of_samples())
    hz_list = wave2hz(wave, frame_rate)
    scales = np.array([estimate_scale(hz) for hz in hz_list])

    return hz_list, scales


# 実行例
# filepath = "../data/きらきら星.wav"
filepath = "../data/カエルの歌.wav"
hz_list, scales = wav2scale(filepath)
plt.plot(hz_list[:-1])
prev_scale = ""
for i, hz_s in enumerate(zip(hz_list, scales)):
    hz = hz_s[0]
    scale = hz_s[1]
    if scale != prev_scale:
        plt.text(x=i, y=hz, s=scale)
    prev_scale = scale

# 経過時間
seq_times = np.arange(0, (len(hz_list) // 10) + 1)
plt.xticks(ticks=np.arange(0, len(hz_list) + 1, 10), labels=seq_times)
plt.xlabel("経過時間")
plt.ylabel("Hz")
plt.savefig("../sample_plot.png")
plt.show()
