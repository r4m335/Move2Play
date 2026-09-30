using UnityEngine;

public class PlayerCombat : MonoBehaviour
{
    public float attackRange = 2f;
    public float attackDamage = 25f;

    private PlayerStats stats; // Reference to PlayerStats

    void Start()
    {
        stats = GetComponent<PlayerStats>();
    }

    void Update()
    {
        // LEFT CLICK → attack
        if (Input.GetMouseButtonDown(0))
        {
            GetComponent<Animator>()?.SetTrigger("Attack");
            DealDamage();
        }

        // RIGHT CLICK → kick
        if (Input.GetMouseButtonDown(1))
        {
            GetComponent<Animator>()?.SetTrigger("Kick");
            DealDamage();
        }

        // Guard input - sets the guard flag on PlayerStats
        if (Input.GetKey(KeyCode.G))
            stats.isGuarding = true;
        else
            stats.isGuarding = false;

        // Sync guard animation
        GetComponent<Animator>()?.SetBool("Guard", stats.isGuarding);
    }

    void DealDamage()
    {
        // Calculate center position in front of the player
        Vector3 center = transform.position + transform.forward * attackRange * 0.5f;

        Collider[] hitEnemies = Physics.OverlapSphere(center, attackRange);

        foreach (Collider enemy in hitEnemies)
        {
            if (enemy.CompareTag("Enemy"))
            {
                EnemyHealth eh = enemy.GetComponent<EnemyHealth>();
                if (eh != null)
                {
                    eh.TakeDamage(attackDamage);
                }
            }
        }
    }
}