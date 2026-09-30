using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;

public class GuideManager : MonoBehaviour
{
    public Image guideImage;
    public Sprite[] guideSprites;

    int currentIndex = 0;

    void Start()
    {
        ShowImage();
    }

    void ShowImage()
    {
        if (guideSprites.Length > 0)
        {
            guideImage.sprite = guideSprites[currentIndex];
        }
    }

    public void NextImage()
    {
        currentIndex++;

        if (currentIndex >= guideSprites.Length)
        {
            currentIndex = 0;
        }

        ShowImage();
    }

    public void PrevImage()
    {
        currentIndex--;

        if (currentIndex < 0)
        {
            currentIndex = guideSprites.Length - 1;
        }

        ShowImage();
    }

    public void BackToMenu()
    {
        SceneManager.LoadScene("MainMenu");
    }
}