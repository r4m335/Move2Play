using UnityEngine;
using UnityEngine.UI;
using TMPro;
using UnityEngine.SceneManagement;
using System.Collections;

public class UIController : MonoBehaviour
{
    [Header("Player & Stats")]
    public PlayerStats player;

    [Header("Game Over")]
    public Image gameOverImage;
    public TMP_Text timeSurvivedText;

    [Header("Victory")]
    public TMP_Text timerText;
    public Image victoryImage;
    public float requiredSurvivalTime = 600f;   // 10 minutes
    public int requiredKills = 15;

    [Header("Objectives")]
    public GameObject objectivePanel;
    public float objectiveDisplayTime = 10f;

    [Header("Audio")]
    public AudioClip victorySound;
    public AudioClip gameOverSound;

    public float FinalSurvivalTime { get; private set; }

    private bool deathTriggered = false;
    private bool victoryTriggered = false;
    private float survivalTimer;

    private Coroutine gameOverCoroutine;
    public bool gameEnded { get; private set; }

    void Start()
    {
        if (gameOverImage != null)
            gameOverImage.gameObject.SetActive(false);
        if (victoryImage != null)
            victoryImage.gameObject.SetActive(false);
        if (timeSurvivedText != null)
            timeSurvivedText.gameObject.SetActive(false);

        if (objectivePanel != null)
        {
            objectivePanel.SetActive(true);
            StartCoroutine(HideObjectivePanel());
        }

        // Reset the AudioManager's gameEnded flag for a new session
        if (AudioManager.Instance != null)
            AudioManager.Instance.ResetGameplayAudio();
    }

    void Update()
    {
        if (player == null) return;

        // Timer while alive and neither game over nor victory has triggered
        if (!deathTriggered && !victoryTriggered)
        {
            survivalTimer += Time.deltaTime;
            float remaining = Mathf.Max(0f, requiredSurvivalTime - survivalTimer);
            int minutes = Mathf.FloorToInt(remaining / 60);
            int seconds = Mathf.FloorToInt(remaining % 60);

            if (timerText != null)
                timerText.text = $"{minutes:00}:{seconds:00}";
        }

        // Victory check – survive the required time AND reach the kill quota
        if (!victoryTriggered &&
            !player.isDead &&
            survivalTimer >= requiredSurvivalTime &&
            KillManager.Instance != null &&
            KillManager.Instance.killCount >= requiredKills)
        {
            TriggerVictory();
        }

        // Time-out failure – time ran out but objectives were not completed
        if (!deathTriggered &&
            !victoryTriggered &&
            survivalTimer >= requiredSurvivalTime &&
            KillManager.Instance != null &&
            KillManager.Instance.killCount < requiredKills)
        {
            TriggerGameOver();
        }

        // ** Change A: Prevent Game Over if Victory has already triggered **
        if (!victoryTriggered &&
            player.isDead &&
            !deathTriggered)
        {
            TriggerGameOver();
        }
    }

    private void TriggerVictory()
    {
        victoryTriggered = true;
        gameEnded = true;
        FinalSurvivalTime = survivalTimer;

        if (victoryImage != null)
            victoryImage.gameObject.SetActive(true);

        // ** Change C: Freeze the game immediately **
        Time.timeScale = 0f;

        // Stop all gameplay audio, then play victory music
        if (AudioManager.Instance != null)
        {
            AudioManager.Instance.StopGameplayAudio();
            if (victorySound != null)
                AudioManager.Instance.PlayMusic(victorySound);
        }
    }

    private void TriggerGameOver()
    {
        // ** Change B: Prevent triggering if Victory already happened **
        if (victoryTriggered)
            return;

        deathTriggered = true;
        gameEnded = true;

        FinalSurvivalTime = survivalTimer;

        if (gameOverImage != null)
            gameOverImage.gameObject.SetActive(true);

        if (timeSurvivedText != null)
        {
            timeSurvivedText.gameObject.SetActive(true);

            int m = Mathf.FloorToInt(FinalSurvivalTime / 60f);
            int s = Mathf.FloorToInt(FinalSurvivalTime % 60f);

            timeSurvivedText.text = $"{m:00}:{s:00}";
        }

        // Stop all gameplay audio, then play game over music
        if (AudioManager.Instance != null)
        {
            AudioManager.Instance.StopGameplayAudio();
            if (gameOverSound != null)
                AudioManager.Instance.PlayMusic(gameOverSound);
        }
    }

    public void CancelGameOverTimer()
    {
        if (gameOverCoroutine != null)
        {
            StopCoroutine(gameOverCoroutine);
            gameOverCoroutine = null;
        }
    }

    private IEnumerator LoadMainMenuAfterDelay(float delay)
    {
        yield return new WaitForSecondsRealtime(delay);
        SceneManager.LoadScene("MainMenu");
    }

    private IEnumerator HideObjectivePanel()
    {
        yield return new WaitForSeconds(objectiveDisplayTime);

        if (objectivePanel != null)
            objectivePanel.SetActive(false);
    }
}