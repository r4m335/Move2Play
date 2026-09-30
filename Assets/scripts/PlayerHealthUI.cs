using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class PlayerHealthUI : MonoBehaviour
{
    public PlayerStats playerStats;

    public Image healthFill;
    public TMP_Text healthText;

    void Update()
    {
        if (playerStats == null) return;

        float percent =
            playerStats.currentHealth /
            playerStats.maxHealth;

        healthFill.fillAmount = percent;

        healthText.text =
            Mathf.RoundToInt(percent * 100f) + "%";
    }
}