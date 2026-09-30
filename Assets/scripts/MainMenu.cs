using UnityEngine;
using UnityEngine.SceneManagement;

public class MainMenu : MonoBehaviour
{
    public string calibrationSceneName = "CalibrationScene";   // mandatory calibration
    public string guideSceneName = "Guide";                   // guide / how-to-play scene

    [Header("Audio")]
    public AudioClip mainMenuMusic;                           // looping main menu track

    void Start()
    {
        // Reset the gameplay audio flag and start the menu music
        if (AudioManager.Instance != null)
        {
            AudioManager.Instance.ResetGameplayAudio();
            AudioManager.Instance.PlayMusic(mainMenuMusic, true);
        }
    }

    public void PlayGame()
    {
        // Reset kill counter for a fresh run
        KillManager.Instance?.ResetKills();

        // Load calibration first – it will then lead to the Game scene
        SceneManager.LoadScene(calibrationSceneName);
    }

    public void LoadGuide()
    {
        SceneManager.LoadScene(guideSceneName);
    }

    public void QuitGame()
    {
        #if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false;
        #else
            Application.Quit();
        #endif
    }
}