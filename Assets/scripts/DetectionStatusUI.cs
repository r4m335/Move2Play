using UnityEngine;
using UnityEngine.UI;

public class DetectionStatusUI : MonoBehaviour
{
    [Header("UI")]
    public Image detectionImage;

    [Header("Sprites")]
    public Sprite detectedSprite;
    public Sprite holdingSprite;
    public Sprite notDetectedSprite;

    public void ShowDetected()
    {
        detectionImage.sprite = detectedSprite;
    }

    public void ShowHolding()
    {
        detectionImage.sprite = holdingSprite;
    }

    public void ShowNotDetected()
    {
        detectionImage.sprite = notDetectedSprite;
    }
}