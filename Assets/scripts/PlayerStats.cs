using UnityEngine;
using UnityEngine.AI;

public class PlayerStats : MonoBehaviour
{
    public float maxHealth = 100f;
    public float currentHealth;
    public float maxEnergy = 100f;
    public float currentEnergy;
    public bool isDead = false;
    public bool isGuarding = false;

    public DamageFlash damageFlash;

    [Header("Audio")]
    public AudioSource breathingSource;   // dedicated source for low-health breathing loop

    private float drownTimer = 0f;
    public bool isInWater = false;
    public float drownDelay = 1.5f;
    public float drownDamagePerSecond = 100f;

    private CombatController combatController;

    void Start()
    {
        currentHealth = maxHealth;
        currentEnergy = maxEnergy;
        combatController = GetComponent<CombatController>();
    }

    void Update()
    {
        if (isDead) return;

        // Drowning logic
        if (isInWater)
        {
            drownTimer += Time.deltaTime;
            if (drownTimer > drownDelay)
            {
                TakeDamage(drownDamagePerSecond * Time.deltaTime);
            }
        }
        else
        {
            drownTimer = 0f;
        }

        // Disable PlayerMovement when in water
        var movement = GetComponent<PlayerMovement>();
        if (movement != null)
            movement.enabled = !isInWater;

        // Low-health breathing (looping AudioSource)
        if (currentHealth <= 25f && !isDead)
        {
            if (breathingSource != null && !breathingSource.isPlaying)
                breathingSource.Play();
        }
        else
        {
            if (breathingSource != null && breathingSource.isPlaying)
                breathingSource.Stop();
        }
    }

    void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Water"))
        {
            isInWater = true;
            Debug.Log("Entered water - Drowning timer started");
        }
    }

    void OnTriggerExit(Collider other)
    {
        if (other.CompareTag("Water"))
        {
            isInWater = false;
            Debug.Log("Exited water - Drowning timer reset");
        }
    }

    public void TakeDamage(float amount)
    {
        if (isDead) return;

        if (isGuarding)
        {
            amount *= 0.5f;

            // Play block impact sound once per hit
            if (combatController != null)
                combatController.PlayBlockImpactSound();
        }

        currentHealth -= amount;

        if (damageFlash != null)
            damageFlash.Flash();

        currentHealth = Mathf.Clamp(currentHealth, 0, maxHealth);

        if (currentHealth <= 0f)
            Die();
    }

    public void Heal(float amount)
    {
        if (isDead) return;
        currentHealth += amount;
        currentHealth = Mathf.Clamp(currentHealth, 0, maxHealth);
        Debug.Log($"Healed {amount}. Health: {currentHealth}/{maxHealth}");
    }

    public void UseEnergy(float amount)
    {
        if (isDead) return;
        currentEnergy -= amount;
        currentEnergy = Mathf.Clamp(currentEnergy, 0, maxEnergy);
    }

    public void RegenEnergy(float amount)
    {
        if (isDead) return;
        currentEnergy += amount;
        currentEnergy = Mathf.Clamp(currentEnergy, 0, maxEnergy);
    }

    void Die()
    {
        if (isDead) return;
        isDead = true;

        // Stop breathing sound on death
        if (breathingSource != null && breathingSource.isPlaying)
            breathingSource.Stop();

        Animator anim = GetComponent<Animator>();
        if (isInWater)
        {
            if (anim != null)
            {
                anim.SetTrigger("Die");
                anim.SetBool("isRunning", false);
                anim.SetBool("Guard", false);
            }
        }
        else
        {
            if (anim != null)
                anim.SetTrigger("Die");
        }

        PlayerMovement movement = GetComponent<PlayerMovement>();
        if (movement != null)
            movement.enabled = false;

        PlayerController controller = GetComponent<PlayerController>();
        if (controller != null)
            controller.enabled = false;

        NavMeshAgent agent = GetComponent<NavMeshAgent>();
        if (agent != null)
        {
            agent.ResetPath();
            agent.enabled = false;
        }

        CharacterController cc = GetComponent<CharacterController>();
        if (cc != null)
            cc.enabled = false;

        Debug.Log("PLAYER DIED");
    }

    public void Respawn(Vector3 spawnPosition)
    {
        currentHealth = maxHealth;
        currentEnergy = maxEnergy;
        isDead = false;
        isGuarding = false;
        isInWater = false;
        drownTimer = 0f;

        // Ensure breathing stops on respawn
        if (breathingSource != null && breathingSource.isPlaying)
            breathingSource.Stop();

        transform.position = spawnPosition;

        var movement = GetComponent<PlayerMovement>();
        if (movement != null) movement.enabled = true;

        NavMeshAgent agent = GetComponent<NavMeshAgent>();
        if (agent != null) agent.enabled = true;

        PlayerController controller = GetComponent<PlayerController>();
        if (controller != null) controller.enabled = true;

        CharacterController cc = GetComponent<CharacterController>();
        if (cc != null) cc.enabled = true;

        GetComponent<Animator>()?.ResetTrigger("Die");
    }
}