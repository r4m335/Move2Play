using UnityEngine;
using UnityEngine.SceneManagement;

public class PauseMenu : MonoBehaviour
{
    [Header("UI References")]
    public GameObject pausePanel;
    public GameObject pauseButton;

    [Header("Gameplay UI")]
    public GameObject killPanel;
    public GameObject timerPanel;
    public GameObject healthPanel;
    public GameObject minimap;
    public GameObject objectivePanel;
    public GameObject enemyTrackerPanel;
    public GameObject medKitTrackerPanel;

    public static bool isPaused = false;

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Escape))
        {
            if (isPaused)
                Resume();
            else
                Pause();
        }
    }

    public void Resume()
    {
        if (pausePanel != null)
            pausePanel.SetActive(false);

        if (pauseButton != null)
            pauseButton.SetActive(true);

        // Show gameplay UI
        if (killPanel != null) killPanel.SetActive(true);
        if (timerPanel != null) timerPanel.SetActive(true);
        if (healthPanel != null) healthPanel.SetActive(true);
        if (minimap != null) minimap.SetActive(true);
        if (enemyTrackerPanel != null) enemyTrackerPanel.SetActive(true);
        if (medKitTrackerPanel != null) medKitTrackerPanel.SetActive(true);

        // Resume gameplay audio
        if (AudioManager.Instance != null)
            AudioManager.Instance.ResumeGameplayAudio();

        Time.timeScale = 1f;
        isPaused = false;

        Cursor.lockState = CursorLockMode.None;
        Cursor.visible = true;
    }

    public void Pause()
    {
        if (pausePanel != null)
            pausePanel.SetActive(true);

        if (pauseButton != null)
            pauseButton.SetActive(false);

        // Hide gameplay UI
        if (killPanel != null) killPanel.SetActive(false);
        if (timerPanel != null) timerPanel.SetActive(false);
        if (healthPanel != null) healthPanel.SetActive(false);
        if (minimap != null) minimap.SetActive(false);
        if (objectivePanel != null) objectivePanel.SetActive(false);
        if (enemyTrackerPanel != null) enemyTrackerPanel.SetActive(false);
        if (medKitTrackerPanel != null) medKitTrackerPanel.SetActive(false);

        // Pause gameplay audio
        if (AudioManager.Instance != null)
            AudioManager.Instance.PauseGameplayAudio();

        Time.timeScale = 0f;
        isPaused = true;

        Cursor.lockState = CursorLockMode.None;
        Cursor.visible = true;
    }

    public void LoadMainMenu()
    {
        StopMediaPipeRunner();

        Time.timeScale = 1f;
        SceneManager.LoadScene("MainMenu");
    }

    public void RestartGame()
    {
        UIController ui = FindObjectOfType<UIController>();
        if (ui != null)
            ui.CancelGameOverTimer();

        Time.timeScale = 1f;
        isPaused = false;

        KillManager.Instance?.ResetKills();

        StopMediaPipeRunner();

        SceneManager.LoadScene(SceneManager.GetActiveScene().name);
    }

    private void StopMediaPipeRunner()
    {
        var runner = FindObjectOfType<Mediapipe.Unity.Sample.PoseLandmarkDetection.PoseLandmarkerRunner>();
        if (runner != null)
        {
            runner.Stop();
            Destroy(runner.gameObject);
            Debug.Log("PauseMenu: Stopped PoseLandmarkerRunner and webcam.");
        }
        else
        {
            WebCamTexture[] webcams = FindObjectsOfType<WebCamTexture>();
            foreach (var webcam in webcams)
            {
                if (webcam.isPlaying)
                {
                    webcam.Stop();
                    Debug.Log("PauseMenu: Stopped WebCamTexture.");
                }
            }
        }
    }
}