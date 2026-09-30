using UnityEngine;

public class ForestAudio : MonoBehaviour
{
    [Header("Looping Ambient")]
    public AudioClip forestAmbient;

    void Start()
    {
        if (AudioManager.Instance == null)
            return;

        // Stop previous scene music
        AudioManager.Instance.musicSource.Stop();

        // Start looping forest ambience
        AudioManager.Instance.PlayAmbient(forestAmbient);
    }
}