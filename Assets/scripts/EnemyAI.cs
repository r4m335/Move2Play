using UnityEngine;
using UnityEngine.AI;

public class EnemyAI : MonoBehaviour
{
    public Transform player;
    public float detectionRange = 10f;
    public float attackRange = 1f;
    public float attackCooldown = 2f;
    public float damageAmount = 3f;
    public float maxChaseTime = 10f;

    // Lose sight buffer
    public float loseSightMultiplier = 1.5f;

    // Wander settings
    public float wanderSpeed = 2f;
    public float chaseSpeed = 5f;
    public float wanderRadius = 10f;
    public float wanderInterval = 3f;

    [Header("Audio")]
    public AudioClip enemyAttackSound;   // played when enemy attacks

    private NavMeshAgent agent;
    private Animator animator;
    private float lastAttackTime;
    private PlayerStats playerStats;
    private EnemyIndicator indicator;
    private EnemyAlertPopup alertPopup;

    private float chaseTimer = 0f;
    private float wanderTimer = 0f;
    private Vector3 wanderTarget;

    private enum State { Idle, Wander, Chase, Attack }
    private State currentState;

    void Start()
    {
        agent = GetComponent<NavMeshAgent>();
        animator = GetComponent<Animator>();
        indicator = GetComponentInChildren<EnemyIndicator>();
        alertPopup = GetComponent<EnemyAlertPopup>();

        if (player != null)
        {
            playerStats = player.GetComponent<PlayerStats>();
            if (playerStats == null)
                Debug.LogError("EnemyAI: Player does not have PlayerStats component!");
        }
        else
        {
            Debug.LogError("EnemyAI: Player reference not assigned!");
        }

        if (!agent.isOnNavMesh)
            Debug.LogError("EnemyAI: Agent is not on NavMesh!");

        if (player != null && playerStats != null && playerStats.currentHealth > 0)
            EnterState(State.Wander);
        else
            EnterState(State.Idle);
    }

    void Update()
    {
        if (player == null || playerStats == null || playerStats.currentHealth <= 0)
        {
            if (currentState != State.Idle)
                EnterState(State.Idle);
            return;
        }

        float distance = Vector3.Distance(transform.position, player.position);

        switch (currentState)
        {
            case State.Idle:
                EnterState(State.Wander);
                break;

            case State.Wander:
                wanderTimer += Time.deltaTime;
                if (wanderTimer >= wanderInterval)
                {
                    SetRandomWanderDestination();
                    wanderTimer = 0f;
                }

                if (agent.remainingDistance <= agent.stoppingDistance)
                {
                    SetRandomWanderDestination();
                    wanderTimer = 0f;
                }

                if (distance < detectionRange)
                    EnterState(State.Chase);
                break;

            case State.Chase:
                chaseTimer += Time.deltaTime;

                if (agent.isOnNavMesh && agent.enabled)
                    agent.SetDestination(player.position);

                if (distance < attackRange)
                {
                    EnterState(State.Attack);
                }
                else if (chaseTimer >= maxChaseTime || distance > detectionRange * loseSightMultiplier)
                {
                    Debug.Log("Enemy gave up chasing.");
                    EnterState(State.Wander);
                }
                break;

            case State.Attack:
                transform.LookAt(player);

                if (Time.time >= lastAttackTime + attackCooldown)
                    PerformAttack();

                if (distance > attackRange)
                    EnterState(State.Chase);
                break;
        }

        UpdateAnimator();
    }

    void EnterState(State newState)
    {
        if (currentState == State.Chase)
            chaseTimer = 0f;

        bool firstDetection = (currentState == State.Wander && newState == State.Chase);

        currentState = newState;

        if (indicator != null)
            indicator.flashing = (newState == State.Chase || newState == State.Attack);

        if (firstDetection && alertPopup != null)
            alertPopup.ShowAlert();

        switch (currentState)
        {
            case State.Idle:
                Debug.Log("Entering Idle state");
                if (agent != null && agent.isOnNavMesh)
                {
                    agent.ResetPath();
                    agent.velocity = Vector3.zero;
                }
                animator.SetTrigger("Idle");
                break;

            case State.Wander:
                agent.speed = wanderSpeed;
                wanderTimer = 0f;
                SetRandomWanderDestination();
                break;

            case State.Chase:
                agent.speed = chaseSpeed;
                chaseTimer = 0f;
                break;

            case State.Attack:
                if (agent != null && agent.isOnNavMesh)
                    agent.ResetPath();
                break;
        }
    }

    void SetRandomWanderDestination()
    {
        if (agent == null || !agent.isOnNavMesh) return;

        Vector3 randomDirection = Random.insideUnitSphere * wanderRadius;
        randomDirection += transform.position;
        if (NavMesh.SamplePosition(randomDirection, out NavMeshHit hit, wanderRadius, NavMesh.AllAreas))
        {
            wanderTarget = hit.position;
            agent.SetDestination(wanderTarget);
        }
    }

    void PerformAttack()
    {
        lastAttackTime = Time.time;
        animator.SetTrigger("Attack");

        // Play enemy attack sound BEFORE dealing damage
        if (enemyAttackSound != null && AudioManager.Instance != null)
            AudioManager.Instance.PlaySFX(enemyAttackSound);

        if (playerStats != null && playerStats.currentHealth > 0)
        {
            playerStats.TakeDamage(damageAmount);
            Debug.Log($"Enemy attacked player for {damageAmount} damage! Player health: {playerStats.currentHealth}/{playerStats.maxHealth}");
        }
        else
        {
            Debug.LogWarning("EnemyAI: Cannot attack - PlayerStats missing or player already dead!");
        }
    }

    void UpdateAnimator()
    {
        if (animator != null && agent != null && agent.isOnNavMesh)
        {
            float speed = agent.velocity.magnitude;
            animator.SetFloat("speed", speed);
        }
    }
}