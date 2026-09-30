using UnityEngine;

public class MedKit : MonoBehaviour
{
    [Range(0f, 1f)]
    public float healPercent = 0.3f;   // 30% of max health

    [Header("Audio")]
    public AudioClip medkitSound;       // played on pickup

    private void OnTriggerEnter(Collider other)
    {
        if (!other.CompareTag("Player"))
            return;

        PlayerStats player = other.GetComponent<PlayerStats>();

        if (player == null)
            return;

        // Don't consume the medkit if the player is already at full health
        if (player.currentHealth >= player.maxHealth)
            return;

        float healAmount = player.maxHealth * healPercent;
        player.Heal(healAmount);

        // Play the pickup sound via the persistent AudioManager (even after Destroy)
        if (medkitSound != null && AudioManager.Instance != null)
            AudioManager.Instance.PlaySFX(medkitSound);

        Destroy(gameObject);
    }
}