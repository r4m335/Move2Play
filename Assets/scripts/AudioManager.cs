using UnityEngine;

public class AudioManager : MonoBehaviour
{
    public static AudioManager Instance;
    public bool gameEnded = false;

    [Header("Audio Sources")]
    public AudioSource musicSource;
    public AudioSource sfxSource;
    public AudioSource ambientSource;
    public AudioSource randomAmbientSource;

    private void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
            DontDestroyOnLoad(gameObject);
        }
        else
        {
            Destroy(gameObject);
        }
    }

    public void ResetGameplayAudio()
    {
        gameEnded = false;
    }

    public void StopGameplayAudio()
    {
        gameEnded = true;
        if (ambientSource != null) ambientSource.Stop();
        if (randomAmbientSource != null) randomAmbientSource.Stop();
        if (sfxSource != null) sfxSource.Stop();
    }

    public void PauseGameplayAudio()
    {
        musicSource.Pause();
        ambientSource.Pause();
        randomAmbientSource.Pause();
        sfxSource.Pause();
    }

    public void ResumeGameplayAudio()
    {
        musicSource.UnPause();
        ambientSource.UnPause();
        randomAmbientSource.UnPause();
        sfxSource.UnPause();
    }

    // --- SFX ---
    public void PlaySFX(AudioClip clip)
    {
        if (gameEnded) return;
        if (clip != null) sfxSource.PlayOneShot(clip);
    }

    public void PlaySFX(AudioClip clip, float volume)
    {
        if (gameEnded) return;
        if (clip != null) sfxSource.PlayOneShot(clip, volume);
    }

    // --- Music ---
    /// <summary>
    /// Play a music clip without looping (used for victory/game over).
    /// </summary>
    public void PlayMusic(AudioClip clip)
    {
        PlayMusic(clip, false);
    }

    /// <summary>
    /// Play a music clip, with optional looping.
    /// Prevents restarting the same track if it's already playing.
    /// </summary>
    public void PlayMusic(AudioClip clip, bool loop)
    {
        if (clip == null) return;

        // Don't restart the same track if it's already playing
        if (musicSource.clip == clip && musicSource.isPlaying)
            return;

        musicSource.Stop();
        musicSource.clip = clip;
        musicSource.loop = loop;
        musicSource.Play();
    }

    // --- Ambient ---
    public void PlayAmbient(AudioClip clip)
    {
        if (gameEnded) return;
        if (clip == null) return;
        ambientSource.clip = clip;
        ambientSource.loop = true;
        ambientSource.Play();
    }

    public void StopAmbient()
    {
        if (ambientSource != null && ambientSource.isPlaying)
            ambientSource.Stop();
    }

    public void PlayRandomAmbient(AudioClip clip)
    {
        if (gameEnded) return;
        if (clip != null) randomAmbientSource.PlayOneShot(clip);
    }

    public void StopAll()
    {
        musicSource.Stop();
        sfxSource.Stop();
        ambientSource.Stop();
        randomAmbientSource.Stop();
    }
}