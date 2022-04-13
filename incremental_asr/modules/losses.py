import torch

def rbkd(teacher: torch.Tensor, student: torch.Tensor, temperature: float=1.0):
    return rbkd_mat(teacher, student, temperature)

def ebkd(teacher_loss: torch.Tensor, student_loss: torch.Tensor, teacher_feature_map: torch.Tensor, student_feature_map: torch.Tensor, temperature: float=1.0):
    return ebkd_mat(teacher_loss, student_loss, teacher_feature_map, student_feature_map, temperature)

def ebkd_mat(teacher_loss: torch.Tensor, student_loss: torch.Tensor, teacher_feature_map: torch.Tensor, student_feature_map: torch.Tensor, temperature: float=1.0):
    teacher_importance_map = torch.autograd.grad(teacher_loss, teacher_feature_map, retain_graph=True)[0]
    student_importance_map = torch.autograd.grad(student_loss, student_feature_map, retain_graph=True)[0]
    
    teacher_attention_map = torch.nn.functional.leaky_relu(teacher_feature_map * teacher_importance_map)
    student_attention_map = torch.nn.functional.leaky_relu(student_feature_map * student_importance_map)
    
    teacher_norm = torch.norm(teacher_attention_map, dim=2)
    student_norm = torch.norm(student_attention_map, dim=2)
    
    with torch.no_grad():
        teacher_norm = teacher_norm.clamp(min=1e-5)
        student_norm = student_norm.clamp(min=1e-5)
    
    loss = (
        torch.norm((teacher_attention_map / teacher_norm.unsqueeze(2))
        - (student_attention_map / student_norm.unsqueeze(2)), dim=2) / teacher_attention_map.shape[1]
    ).sum(dim=1).mean()
    
    return loss

def ebkd_loop(teacher_loss: torch.Tensor, student_loss: torch.Tensor, teacher_feature_map: torch.Tensor, student_feature_map: torch.Tensor, temperature: float=1.0):
    teacher_importance_map = torch.autograd.grad(teacher_loss, teacher_feature_map, retain_graph=True)[0]
    student_importance_map = torch.autograd.grad(student_loss, student_feature_map, retain_graph=True)[0]
    
    teacher_attention_map = torch.nn.functional.leaky_relu(teacher_feature_map * teacher_importance_map)
    student_attention_map = torch.nn.functional.leaky_relu(student_feature_map * student_importance_map)
    losses = []
    for b in range(teacher_attention_map.shape[0]):
        sum = 0
        for k in range(teacher_attention_map.shape[1]):
            t_norm = torch.norm(teacher_attention_map[b, k, :])
            s_norm = torch.norm(student_attention_map[b, k, :])
            
            norm = (student_attention_map[b, k, :] / s_norm) - (teacher_attention_map[b, k, :] / t_norm)
            # print('LOOP:', norm.shape)
            norm = torch.norm(norm)
            sum += norm
        sum = sum / teacher_attention_map.shape[1]
        losses.append(sum)
    # print(losses)
    
    final_loss = torch.Tensor(losses).mean()
    print('LOOP:', final_loss)
            

def rbkd_mat(teacher: torch.Tensor, student: torch.Tensor, temperature: float=1.0):   
    teacher_sl = torch.div((torch.exp(teacher / temperature)).permute(2,0,1), (torch.exp(teacher / temperature)).sum(dim=2)).permute(1,2,0)
    student_sl = torch.div((torch.exp(student / temperature)).permute(2,0,1), (torch.exp(student / temperature)).sum(dim=2)).permute(1,2,0)
    
    loss = -(teacher_sl * torch.log(student_sl)).sum(dim=2).sum(dim=1)
    return loss.mean()

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