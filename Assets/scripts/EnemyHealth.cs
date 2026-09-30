using UnityEngine;
using UnityEngine.AI;

public class EnemyHealth : MonoBehaviour
{
    public float maxHealth = 100f;
    public float currentHealth;

    [Header("Audio")]
    public AudioClip enemyHitSound;    // played when enemy is damaged but not killed
    public AudioClip enemyDeathSound;  // played when enemy dies

    private Animator animator;
    private bool isDead = false;
    private NavMeshAgent agent;
    private Collider col;
    private EnemyAI enemyAI;

    void Start()
    {
        currentHealth = maxHealth;
        animator = GetComponent<Animator>();
        agent = GetComponent<NavMeshAgent>();
        col = GetComponent<Collider>();
        enemyAI = GetComponent<EnemyAI>();
    }

    public void TakeDamage(float amount)
    {
        if (isDead) return;

        currentHealth -= amount;
        currentHealth = Mathf.Clamp(currentHealth, 0, maxHealth);

        if (currentHealth <= 0)
        {
            Die(); // death sound plays inside Die()
        }
        else
        {
            // Play hurt sound only if the enemy survives the hit
            if (enemyHitSound != null && AudioManager.Instance != null)
                AudioManager.Instance.PlaySFX(enemyHitSound);
        }
    }

    void Die()
    {
        isDead = true;

        // Play death sound BEFORE disabling components or destroying the object
        if (enemyDeathSound != null && AudioManager.Instance != null)
            AudioManager.Instance.PlaySFX(enemyDeathSound);

        if (enemyAI != null)
            enemyAI.enabled = false;

        animator.SetTrigger("Die");

        if (agent != null)
        {
            agent.ResetPath();
            agent.enabled = false;
        }

        if (col != null)
            col.enabled = false;

        KillManager.Instance?.AddKill();

        // Destroy after delay – death sound continues via AudioManager
        Destroy(gameObject, 3f);
    }
}