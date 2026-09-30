using UnityEngine;
using System;

public class CombatController : MonoBehaviour
{
    [Header("References")]
    public Animator animator;
    public PlayerStats playerStats;

    [Header("Combat Settings")]
    public float attackRange = 2f;
    public float attackDamage = 25f;
    public float blockDuration = 1.5f;
    [Tooltip("Cooldown in seconds between consecutive attacks (punch/kick)")]
    public float attackCooldown = 0.3f;

    [Header("Audio")]
    public AudioClip attackSound;
    public AudioClip blockStartSound;    // sound when block is raised
    public AudioClip blockImpactSound;   // sound when a hit is blocked

    [Header("Debug")]
    public bool logCombat = true;

    public Action OnBlockEnd;

    private bool isBlocking = false;
    private float blockEndTime = 0f;
    private float lastAttackTime = 0f;

    void Awake()
    {
        if (animator == null)
            animator = GetComponent<Animator>();

        if (animator == null)
            Debug.LogError("[CombatController] Animator not found!");

        if (playerStats == null)
            playerStats = GetComponent<PlayerStats>();
    }

    void Update()
    {
        // (if needed, handle block duration timers here)
    }

    #region Public Action Methods
    public void Punch()
    {
        if (animator == null) return;

        if (isBlocking)
        {
            if (logCombat) Debug.Log("[CombatController] Cannot punch while blocking");
            return;
        }

        if (Time.time - lastAttackTime < attackCooldown) return;

        lastAttackTime = Time.time;
        if (logCombat) Debug.Log("[CombatController] PUNCH TRIGGERED");
        animator.SetTrigger("Attack");

        if (attackSound != null && AudioManager.Instance != null)
            AudioManager.Instance.PlaySFX(attackSound);

        DealDamage();
    }

    public void Kick()
    {
        if (animator == null) return;

        if (isBlocking)
        {
            if (logCombat) Debug.Log("[CombatController] Cannot kick while blocking");
            return;
        }

        if (Time.time - lastAttackTime < attackCooldown) return;

        lastAttackTime = Time.time;
        if (logCombat) Debug.Log("[CombatController] KICK TRIGGERED");
        animator.SetTrigger("Kick");

        if (attackSound != null && AudioManager.Instance != null)
            AudioManager.Instance.PlaySFX(attackSound);

        DealDamage();
    }

    public void Block()
    {
        if (animator == null) return;

        if (IsAttacking())
        {
            if (logCombat) Debug.Log("[CombatController] Cannot block while attacking");
            return;
        }

        if (isBlocking)
        {
            if (logCombat) Debug.Log("[CombatController] BLOCK REFRESHED");
            return;
        }

        if (logCombat) Debug.Log("[CombatController] BLOCK STARTED");

        // Play block start sound (only when guard is raised)
        if (blockStartSound != null && AudioManager.Instance != null)
            AudioManager.Instance.PlaySFX(blockStartSound);

        animator.SetBool("Guard", true);
        isBlocking = true;

        if (playerStats != null)
            playerStats.isGuarding = true;
    }

    public void StopBlocking()
    {
        if (!isBlocking) return;

        if (logCombat) Debug.Log("[CombatController] BLOCK STOPPED");

        if (animator != null)
            animator.SetBool("Guard", false);

        isBlocking = false;

        if (playerStats != null)
            playerStats.isGuarding = false;

        OnBlockEnd?.Invoke();
    }

    /// <summary>
    /// Called from PlayerStats when an attack hits the guard.
    /// Plays the block impact sound (clash) once per hit.
    /// </summary>
    public void PlayBlockImpactSound()
    {
        if (blockImpactSound != null && AudioManager.Instance != null)
            AudioManager.Instance.PlaySFX(blockImpactSound);
    }
    #endregion

    #region Damage Dealing
    public void DealDamage()
    {
        if (logCombat) Debug.Log($"[CombatController] DealDamage() called. Range={attackRange}, Damage={attackDamage}");

        Vector3 center = transform.position + transform.forward * (attackRange * 0.5f);
        Collider[] hitEnemies = Physics.OverlapSphere(center, attackRange);

        bool hitAny = false;
        foreach (Collider enemy in hitEnemies)
        {
            if (enemy.CompareTag("Enemy"))
            {
                EnemyHealth eh = enemy.GetComponent<EnemyHealth>();
                if (eh != null)
                {
                    eh.TakeDamage(attackDamage);
                    hitAny = true;
                    if (logCombat) Debug.Log($"[CombatController] Hit {enemy.name} for {attackDamage} damage");
                }
            }
        }

        if (!hitAny && logCombat)
            Debug.Log("[CombatController] No enemies hit.");
    }
    #endregion

    #region Public Properties & Helpers
    public bool IsBlocking() => isBlocking;
    public bool IsAttacking()
    {
        if (animator == null) return false;
        AnimatorStateInfo stateInfo = animator.GetCurrentAnimatorStateInfo(0);
        return stateInfo.IsTag("Attack");
    }
    #endregion

    private void OnDrawGizmosSelected()
    {
        if (Application.isPlaying)
        {
            Gizmos.color = Color.red;
            Vector3 center = transform.position + transform.forward * (attackRange * 0.5f);
            Gizmos.DrawWireSphere(center, attackRange);
        }
    }
}