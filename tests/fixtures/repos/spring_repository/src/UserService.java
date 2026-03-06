package com.example.repository;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Service
public class UserService {

    private final UserRepository userRepository;

    public UserService(UserRepository userRepository) {
        this.userRepository = userRepository;
    }

    public User findById(Long id) {
        return userRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("User not found: " + id));
    }

    @Transactional
    public User createUser(String email, String name) {
        if (userRepository.existsByEmail(email)) {
            throw new RuntimeException("Email already registered: " + email);
        }
        return userRepository.save(new User(email, name));
    }

    public List<User> findActiveUsers() {
        return userRepository.findByActiveTrue();
    }

    @Transactional
    public void deactivateUser(Long id) {
        User user = findById(id);
        user.deactivate();
        userRepository.save(user);
    }
}
