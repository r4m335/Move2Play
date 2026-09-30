using UnityEngine;
using UnityEngine.UI;

public class DamageFlash : MonoBehaviour
{
    public Image flashImage;

    public float flashAlpha = 1f;       // Peak alpha (1 = fully opaque)
    public float holdDuration = 0.15f;  // How long to stay fully visible
    public float fadeSpeed = 2f;        // Alpha decrease per second

    private float currentAlpha = 0f;
    private float holdTimer = 0f;

    void Start()
    {
        // Start fully transparent and hidden
        currentAlpha = 0f;
        holdTimer = 0f;
        if (flashImage != null)
        {
            Color c = flashImage.color;
            c.a = 0f;
            flashImage.color = c;
            flashImage.enabled = false;
        }
    }

    void Update()
    {
        if (flashImage == null) return;

        // If we're still in the hold phase, just wait
        if (holdTimer > 0f)
        {
            holdTimer -= Time.deltaTime;
            return; // don't fade yet
        }

        // After hold, fade out
        if (currentAlpha > 0f)
        {
            currentAlpha = Mathf.MoveTowards(
                currentAlpha,
                0f,
                fadeSpeed * Time.deltaTime);

            Color c = flashImage.color;
            c.a = currentAlpha;
            flashImage.color = c;

            // Hide completely when fade finishes
            if (currentAlpha <= 0f)
            {
                flashImage.enabled = false;
            }
        }
    }

    public void Flash()
    {
        if (flashImage == null) return;

        // Show instantly at peak alpha
        flashImage.enabled = true;
        currentAlpha = flashAlpha;
        holdTimer = holdDuration;

        // Set color to fully opaque
        Color c = flashImage.color;
        c.a = flashAlpha;
        flashImage.color = c;
    }
}