using UnityEngine;
using UnityEngine.UI;

public class EnemyHealthBar : MonoBehaviour
{
    public Image fillImage;
    public Canvas canvas;          // Assign the Canvas component (not the GameObject)

    Camera mainCam;

    void Start()
    {
        canvas.enabled = false;    // Hidden at full health
        mainCam = Camera.main;
    }

    void LateUpdate()
    {
        if (mainCam != null)
            transform.forward = mainCam.transform.forward;
    }

    // Call this to update the fill (does NOT affect visibility)
    public void SetHealth(float current, float max)
    {
        fillImage.fillAmount = current / max;
    }

    // Call this exactly once – when the enemy takes its first damage
    public void Show()
    {
        canvas.enabled = true;
    }
}