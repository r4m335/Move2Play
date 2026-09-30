using UnityEngine;
using UnityEngine.UI;

public class RunLockImageDisplay : MonoBehaviour
{
    public PlayerController player;
    public Image runLockImage;

    private bool previousRunLockState = false;
    private float displayTimer = 0f;

    public float displayDuration = 5f;

    void Start()
    {
        if (runLockImage != null)
            runLockImage.enabled = false;
    }

    void Update()
    {
        if (player == null || runLockImage == null)
            return;

        // Detect when run lock first activates
        if (player.IsRunLocked && !previousRunLockState)
        {
            runLockImage.enabled = true;
            displayTimer = displayDuration;
        }

        // Countdown
        if (displayTimer > 0)
        {
            displayTimer -= Time.deltaTime;

            if (displayTimer <= 0)
            {
                runLockImage.enabled = false;
            }
        }

        previousRunLockState = player.IsRunLocked;
    }
}