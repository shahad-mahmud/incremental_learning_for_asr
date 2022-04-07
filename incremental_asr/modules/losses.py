import torch

def rbkd(teacher, student, temperature: float=1.0):
    return rbkd_mat(teacher, student, temperature)

def rbkd_mat(teacher: torch.Tensor, student: torch.Tensor, temperature: float=1.0):   
    teacher_sl = torch.div((teacher ** (1 / temperature)).permute(2,0,1), (teacher ** (1 / temperature)).sum(dim=2)).permute(1,2,0)
    student_sl = torch.div((student ** (1 / temperature)).permute(2,0,1), (student ** (1 / temperature)).sum(dim=2)).permute(1,2,0)
    
    loss = -(teacher_sl * torch.log(student_sl)).sum(dim=2).sum(dim=1).mean()
    return loss

def rbkd_loop(teacher: torch.Tensor, student: torch.Tensor, temperature: float=1.0):
    batch_size = teacher.shape[0]
    sequence_length = student.shape[1]
    vocab_size = teacher.shape[2]
    
    losses = []
    for sequence in range(batch_size):
        loss = 0
        for k in range(sequence_length):
            temp_loss = 0
            for m in range(vocab_size):
                p_t_i_j = 0
                for m_ in range(vocab_size):
                    p_t_i_j += teacher[sequence, k, m_] ** (1 / temperature)
                p_t_i_j = teacher[sequence, k, m] ** (1 / temperature) / p_t_i_j
                
                p_s_i_j = 0
                for m_ in range(vocab_size):
                    p_s_i_j += student[sequence, k, m_] ** (1 / temperature)
                p_s_i_j = student[sequence, k, m] ** (1 / temperature) / p_s_i_j
                
                p = p_t_i_j * torch.log(p_s_i_j)
                temp_loss += p
            loss += temp_loss
        losses.append(-loss)
    
    final_loss = torch.Tensor(losses).mean()
    return final_loss